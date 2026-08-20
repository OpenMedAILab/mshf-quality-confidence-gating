#!/usr/bin/env python3
"""Train the protocol MSHF quality encoder and run frozen DeepDRiD inference."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from train_deepdrid_protocol_model import FundusBorderCrop


DIMENSIONS = ("illumination", "clarity", "contrast", "overall")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_one(item):
    image_id, source_path, destination_path, cache_size = item
    destination = Path(destination_path)
    if destination.exists():
        return image_id, str(destination), False
    with Image.open(source_path) as source:
        image = FundusBorderCrop()(source.convert("RGB"))
        image = image.resize((cache_size, cache_size), resample=Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="JPEG", quality=95, subsampling=0)
    return image_id, str(destination), True


def prepare_cache(frame, cache_directory, cache_size, workers):
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    items = [
        (str(row.image_id), str(row.image_path), str(cache_directory / f"{row.image_id}.jpg"), cache_size)
        for row in frame.itertuples(index=False)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(cache_one, items))
    path_map = {image_id: path for image_id, path, _ in results}
    cached = frame.copy()
    cached["original_image_path"] = cached["image_path"]
    cached["image_path"] = cached["image_id"].astype(str).map(path_map)
    print(
        json.dumps(
            {
                "cache": str(cache_directory),
                "created": int(sum(created for _, _, created in results)),
                "reused": int(sum(not created for _, _, created in results)),
            }
        ),
        flush=True,
    )
    return cached


class QualityDataset(Dataset):
    def __init__(self, frame, transform, include_labels=True):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.include_labels = include_labels

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        if self.include_labels:
            labels = torch.tensor([float(row[f"{d}_majority"]) for d in DIMENSIONS], dtype=torch.float32)
        else:
            labels = torch.zeros(len(DIMENSIONS), dtype=torch.float32)
        return self.transform(image), labels, str(row["image_id"])


def make_transforms(image_size):
    normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def make_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(DIMENSIONS))
    return model


def make_loader(frame, transform, batch_size, workers, shuffle, include_labels=True):
    return DataLoader(
        QualityDataset(frame, transform, include_labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def train_fixed(model, loader, device, args, run_name):
    head_parameters = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.head_lr * 0.01
    )
    loss_function = nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = loss_function(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row = {"run": run_name, "epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    amp_enabled = device.type == "cuda"
    rows = []
    for images, labels, image_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            probabilities = torch.sigmoid(model(images)).float().cpu().numpy()
        labels = labels.numpy()
        for image_id, truth, probability in zip(image_ids, labels, probabilities):
            row = {"image_id": image_id}
            for dimension, target, score in zip(DIMENSIONS, truth, probability):
                row[f"{dimension}_true"] = float(target)
                row[f"q_{dimension}"] = float(score)
            rows.append(row)
    return pd.DataFrame(rows)


def quality_metrics(predictions):
    output = {}
    for dimension in DIMENSIONS:
        y = predictions[f"{dimension}_true"].to_numpy(dtype=int)
        p = predictions[f"q_{dimension}"].to_numpy(dtype=float)
        output[dimension] = {
            "n": int(len(y)),
            "positive_rate": float(np.mean(y)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, p)),
        }
    return output


def run_loso(mshf, train_transform, evaluation_transform, args, device):
    output_directory = Path(args.loso_output)
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = []
    histories = []
    source_metrics = {}
    for source_index, held_source in enumerate(sorted(mshf["source"].unique()), start=1):
        seed_all(args.seed + source_index)
        fit_frame = mshf[mshf["source"] != held_source].copy()
        held_frame = mshf[mshf["source"] == held_source].copy()
        train_loader = make_loader(fit_frame, train_transform, args.batch_size, args.workers, True)
        held_loader = make_loader(held_frame, evaluation_transform, args.batch_size, args.workers, False)
        model = make_model().to(device)
        histories.extend(train_fixed(model, train_loader, device, args, f"loso_{held_source}"))
        held_predictions = predict(model, held_loader, device)
        held_predictions["source"] = held_source
        predictions.append(held_predictions)
        source_metrics[held_source] = quality_metrics(held_predictions)
        print(json.dumps({"held_source": held_source, "metrics": source_metrics[held_source]}), flush=True)
        del model
        torch.cuda.empty_cache()
    pooled = pd.concat(predictions, ignore_index=True)
    summary = {"pooled": quality_metrics(pooled), "by_source": source_metrics, "history": histories}
    pooled.to_csv(output_directory / "mshf_loso_predictions.csv", index=False)
    (output_directory / "mshf_loso_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_final_and_infer(mshf, deepdrid, train_transform, evaluation_transform, args, device):
    quality_output = Path(args.final_output)
    quality_output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed + 100)
    train_loader = make_loader(mshf, train_transform, args.batch_size, args.workers, True)
    model = make_model().to(device)
    history = train_fixed(model, train_loader, device, args, "final_all_mshf")
    checkpoint_path = quality_output / "final_mshf_quality_convnext_tiny.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": "convnext_tiny",
            "dims": list(DIMENSIONS),
            "image_size": args.image_size,
            "pretraining": "torchvision ImageNet DEFAULT",
            "epochs": args.epochs,
            "args": vars(args),
        },
        checkpoint_path,
    )
    (quality_output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    inference_loader = make_loader(
        deepdrid, evaluation_transform, args.batch_size, args.workers, False, include_labels=False
    )
    predictions = predict(model, inference_loader, device)
    predictions = predictions.drop(columns=[f"{dimension}_true" for dimension in DIMENSIONS])
    path_lookup = deepdrid.set_index("image_id")["original_image_path"].to_dict()
    predictions["image_name"] = predictions["image_id"].astype(str) + ".jpg"
    predictions["image_path"] = predictions["image_id"].map(path_lookup)
    predictions = predictions[["image_name", "image_path", *[f"q_{d}" for d in DIMENSIONS]]]
    deepdrid_output = Path(args.deepdrid_output)
    deepdrid_output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(deepdrid_output / "deepdrid_quality_predictions.csv", index=False)
    return {"checkpoint": str(checkpoint_path), "deepdrid_predictions": int(len(predictions)), "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mshf-manifest", default="research_mshf/outputs/manifests/mshf_manifest.csv")
    parser.add_argument("--mshf-root", default="MSHF_dataset_2.0/MSHF dataset 2.0/Original")
    parser.add_argument("--deepdrid-manifest", default="research_mshf/outputs/manifests/deepdrid_regular_manifest.csv")
    parser.add_argument("--mshf-cache", default="research_mshf/outputs/cache/mshf_fundus_448")
    parser.add_argument("--deepdrid-cache", default="research_mshf/outputs/cache/deepdrid_fundus_448")
    parser.add_argument("--loso-output", default="research_mshf/outputs/mshf_quality_loso_convnext_pretrained")
    parser.add_argument("--final-output", default="research_mshf/outputs/mshf_quality_convnext_pretrained")
    parser.add_argument("--deepdrid-output", default="research_mshf/outputs/deepdrid_quality_convnext_pretrained")
    parser.add_argument("--mode", choices=("loso", "final", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--cache-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mshf = pd.read_csv(args.mshf_manifest).dropna(
        subset=["relative_path", "source", *[f"{dimension}_majority" for dimension in DIMENSIONS]]
    )
    mshf["image_id"] = mshf["image_name"].astype(str).str.replace(".jpg", "", regex=False)
    mshf["image_path"] = mshf["relative_path"].map(lambda value: str(Path(args.mshf_root) / value))
    mshf = prepare_cache(mshf, args.mshf_cache, args.cache_size, args.workers)

    deepdrid = pd.read_csv(args.deepdrid_manifest).dropna(subset=["image_id", "image_path"])
    deepdrid = prepare_cache(deepdrid, args.deepdrid_cache, args.cache_size, args.workers)
    train_transform, evaluation_transform = make_transforms(args.image_size)
    configuration = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "mshf_images": int(len(mshf)),
        "mshf_sources": mshf["source"].value_counts().to_dict(),
        "deepdrid_images": int(len(deepdrid)),
        "args": vars(args),
    }
    print(json.dumps(configuration), flush=True)

    summaries = {}
    if args.mode in {"loso", "both"}:
        summaries["loso"] = run_loso(mshf, train_transform, evaluation_transform, args, device)
    if args.mode in {"final", "both"}:
        summaries["final"] = run_final_and_infer(
            mshf, deepdrid, train_transform, evaluation_transform, args, device
        )
    for directory in (Path(args.loso_output), Path(args.final_output), Path(args.deepdrid_output)):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(json.dumps({"completed": list(summaries), "device": str(device)}), flush=True)


if __name__ == "__main__":
    main()
