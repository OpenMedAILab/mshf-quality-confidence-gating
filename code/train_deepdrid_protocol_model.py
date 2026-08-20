#!/usr/bin/env python3
"""Train the protocol disease model with patient-grouped cross-fitting.

The schedule is fixed before training. Validation predictions are generated only
after the final all-training-data model has completed the same fixed schedule.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageChops
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logit(values):
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(values):
    z = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


class FundusBorderCrop:
    def __init__(self, threshold=10, padding_fraction=0.02):
        self.threshold = threshold
        self.padding_fraction = padding_fraction

    def __call__(self, image):
        red, green, blue = image.split()
        maximum_channel = ImageChops.lighter(ImageChops.lighter(red, green), blue)
        thresholded = maximum_channel.point(lambda value: 255 if value > self.threshold else 0)
        bounding_box = thresholded.getbbox()
        if bounding_box is None:
            return image
        x0, y0, x1, y1 = bounding_box
        padding = int(max(y1 - y0, x1 - x0) * self.padding_fraction)
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(image.width, x1 + padding)
        y1 = min(image.height, y1 + padding)
        return image.crop((x0, y0, x1, y1))


class FundusDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        label = torch.tensor(float(row["referable_dr"]), dtype=torch.float32)
        return self.transform(image), label, str(row["image_id"])


def make_transforms(image_size, inputs_are_precropped=False):
    normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    border_crop = [] if inputs_are_precropped else [FundusBorderCrop()]
    train_transform = transforms.Compose(
        border_crop
        + [
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
        border_crop
        + [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def cache_one_fundus(item):
    image_id, source_path, destination_path, cache_size = item
    destination = Path(destination_path)
    if destination.exists():
        return image_id, str(destination), False
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        image = FundusBorderCrop()(image)
        image = image.resize((cache_size, cache_size), resample=Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="JPEG", quality=95, subsampling=0)
    return image_id, str(destination), True


def prepare_fundus_cache(frame, cache_directory, cache_size, workers):
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    items = [
        (
            str(row.image_id),
            str(row.image_path),
            str(cache_directory / f"{row.image_id}.jpg"),
            cache_size,
        )
        for row in frame.itertuples(index=False)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(cache_one_fundus, items))
    path_map = {image_id: path for image_id, path, _ in results}
    cached = frame.copy()
    cached["original_image_path"] = cached["image_path"]
    cached["image_path"] = cached["image_id"].astype(str).map(path_map)
    print(
        json.dumps(
            {
                "preprocess_cache": str(cache_directory),
                "cache_size": cache_size,
                "created": int(sum(created for _, _, created in results)),
                "reused": int(sum(not created for _, _, created in results)),
            }
        ),
        flush=True,
    )
    return cached


def make_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def make_loader(frame, transform, batch_size, workers, shuffle):
    return DataLoader(
        FundusDataset(frame, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def make_optimizer(model, head_learning_rate, backbone_learning_rate, weight_decay):
    head_parameters = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_learning_rate},
            {"params": head_parameters, "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )


def train_fixed_schedule(model, loader, device, epochs, head_lr, backbone_lr, weight_decay, run_name):
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = make_optimizer(model, head_lr, backbone_lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=head_lr * 0.01)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images).squeeze(1)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row = {
            "run": run_name,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "backbone_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    rows = []
    amp_enabled = device.type == "cuda"
    for images, labels, image_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()
        for image_id, label, probability in zip(image_ids, labels.numpy(), probabilities):
            rows.append({"image_id": image_id, "y_true": int(label), "disease_prob": float(probability)})
    return pd.DataFrame(rows)


def binary_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "error_rate_05": float(np.mean((p >= 0.5).astype(int) != y)),
    }


def eye_metrics(predictions, metadata):
    merged = predictions.merge(
        metadata[["image_id", "patient_id", "eye", "dr_grade", "referable_dr"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    merged["disease_logit"] = logit(merged["disease_prob"])
    eyes = (
        merged.groupby(["patient_id", "eye"], observed=True)
        .agg(y_true=("y_true", "first"), disease_logit=("disease_logit", "mean"), n_views=("image_id", "size"))
        .reset_index()
    )
    eyes["disease_prob"] = sigmoid(eyes["disease_logit"])
    return binary_metrics(eyes["y_true"], eyes["disease_prob"])


def run_oof(train, train_transform, evaluation_transform, args, device, output_dir):
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    prediction_parts = []
    history = []
    fold_metrics = []
    for fold, (fit_index, holdout_index) in enumerate(
        splitter.split(train, train["referable_dr"], groups=train["patient_id"]), start=1
    ):
        seed_all(args.seed + fold)
        fit_frame = train.iloc[fit_index].copy()
        holdout_frame = train.iloc[holdout_index].copy()
        train_loader = make_loader(fit_frame, train_transform, args.batch_size, args.workers, True)
        holdout_loader = make_loader(holdout_frame, evaluation_transform, args.batch_size, args.workers, False)
        model = make_model().to(device)
        history.extend(
            train_fixed_schedule(
                model,
                train_loader,
                device,
                args.epochs,
                args.head_lr,
                args.backbone_lr,
                args.weight_decay,
                f"fold_{fold}",
            )
        )
        fold_predictions = predict(model, holdout_loader, device)
        fold_predictions["fold"] = fold
        prediction_parts.append(fold_predictions)
        metrics = {
            "fold": fold,
            "images": binary_metrics(fold_predictions["y_true"], fold_predictions["disease_prob"]),
            "eyes": eye_metrics(fold_predictions, train),
        }
        fold_metrics.append(metrics)
        print(json.dumps(metrics), flush=True)
        del model
        torch.cuda.empty_cache()

    oof = pd.concat(prediction_parts, ignore_index=True)
    oof = oof.merge(
        train[["image_id", "patient_id", "dr_grade", "referable_dr"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    summary = {
        "images": binary_metrics(oof["y_true"], oof["disease_prob"]),
        "eyes": eye_metrics(oof[["image_id", "y_true", "disease_prob"]], train),
        "folds": fold_metrics,
        "history": history,
    }
    oof.to_csv(output_dir / "train_oof_predictions.csv", index=False)
    (output_dir / "oof_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_final(train, validation, train_transform, evaluation_transform, args, device, output_dir):
    seed_all(args.seed + 100)
    train_loader = make_loader(train, train_transform, args.batch_size, args.workers, True)
    validation_loader = make_loader(validation, evaluation_transform, args.batch_size, args.workers, False)
    model = make_model().to(device)
    history = train_fixed_schedule(
        model,
        train_loader,
        device,
        args.epochs,
        args.head_lr,
        args.backbone_lr,
        args.weight_decay,
        "final_all_training_data",
    )
    validation_predictions = predict(model, validation_loader, device)
    metrics = {
        "images": binary_metrics(validation_predictions["y_true"], validation_predictions["disease_prob"]),
        "eyes": eye_metrics(validation_predictions, validation),
        "history": history,
        "validation_used_for_model_selection": False,
    }
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "convnext_tiny",
            "pretraining": "torchvision ImageNet DEFAULT",
            "image_size": args.image_size,
            "epochs": args.epochs,
            "args": vars(args),
        },
        output_dir / "final_convnext_tiny.pt",
    )
    (output_dir / "deepdrid_dr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"final_validation": metrics}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="research_mshf/outputs/manifests/deepdrid_regular_manifest.csv")
    parser.add_argument("--oof-output", default="research_mshf/outputs/deepdrid_oof_convnext_pretrained")
    parser.add_argument("--final-output", default="research_mshf/outputs/deepdrid_dr_convnext_pretrained")
    parser.add_argument("--mode", choices=("oof", "final", "both"), default="both")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preprocess-cache", default="research_mshf/outputs/cache/deepdrid_fundus_448")
    parser.add_argument("--cache-size", type=int, default=448)
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
    manifest = pd.read_csv(args.manifest).dropna(subset=["referable_dr", "image_path"]).copy()
    if args.preprocess_cache:
        manifest = prepare_fundus_cache(manifest, args.preprocess_cache, args.cache_size, args.workers)
    train = manifest[manifest["split"] == "train"].copy()
    validation = manifest[manifest["split"] == "validation"].copy()
    train_transform, evaluation_transform = make_transforms(
        args.image_size, inputs_are_precropped=bool(args.preprocess_cache)
    )
    configuration = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_images": int(len(train)),
        "train_patients": int(train["patient_id"].nunique()),
        "validation_images": int(len(validation)),
        "validation_patients": int(validation["patient_id"].nunique()),
        "args": vars(args),
    }
    print(json.dumps(configuration), flush=True)

    summaries = {}
    if args.mode in {"oof", "both"}:
        oof_output = Path(args.oof_output)
        oof_output.mkdir(parents=True, exist_ok=True)
        (oof_output / "run_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
        summaries["oof"] = run_oof(train, train_transform, evaluation_transform, args, device, oof_output)
    if args.mode in {"final", "both"}:
        final_output = Path(args.final_output)
        final_output.mkdir(parents=True, exist_ok=True)
        (final_output / "run_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
        summaries["final"] = run_final(
            train, validation, train_transform, evaluation_transform, args, device, final_output
        )
    print(json.dumps({"completed": list(summaries), "device": str(device)}), flush=True)


if __name__ == "__main__":
    main()
