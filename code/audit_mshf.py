#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

QUALITY_DIMS = ["illumination", "clarity", "contrast", "overall"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SOURCE_PATTERNS = [
    ("UWF-mosaic", re.compile(r"^UWF-mosaic-", re.I)),
    ("DR-XJU", re.compile(r"^DR-XJU-", re.I)),
    ("DR-ZJU", re.compile(r"^DR-ZJU-", re.I)),
    ("Glaucoma", re.compile(r"^Glaucoma-", re.I)),
    ("Healthy", re.compile(r"^Healthy-", re.I)),
    ("Local1", re.compile(r"^Local1-", re.I)),
    ("Local2", re.compile(r"^Local2-", re.I)),
]

def infer_source(name: str) -> str:
    for source, pattern in SOURCE_PATTERNS:
        if pattern.search(name):
            return source
    return "unknown"

def norm_dim(x: str) -> str:
    x = str(x).strip().lower()
    return "overall" if x == "overall" else x

def as_binary(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return np.nan
    try:
        f = float(v)
    except Exception:
        return np.nan
    return int(f) if f in (0.0, 1.0) else np.nan

def scan_images(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rows.append({"image_name": p.name, "source_from_name": infer_source(p.name), "relative_path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size})
    return pd.DataFrame(rows)

def read_individual_labels(path: Path) -> Tuple[pd.DataFrame, Dict]:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row = None
    dim_row = None
    for i in range(min(10, len(raw) - 1)):
        vals0 = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        vals1 = [str(v).strip().lower() for v in raw.iloc[i + 1].tolist()]
        if "image name" in vals0 and "illumination" in vals1 and "overall" in vals1:
            header_row = i
            dim_row = i + 1
            break
    if header_row is None or dim_row is None:
        raise RuntimeError(f"Could not locate two-line header in {path}")
    header0 = [str(v).strip().lower() for v in raw.iloc[header_row].tolist()]
    header1 = [norm_dim(v) for v in raw.iloc[dim_row].tolist()]
    image_col = header0.index("image name")
    header = []
    for a, b in zip(header0, header1):
        header.append("image name" if a == "image name" else b)
    rows = []
    for _, r in raw.iloc[dim_row + 1:].iterrows():
        name = r.iloc[image_col]
        if pd.isna(name):
            continue
        name = str(name).strip()
        row = {"image_name": name, "source_from_label": infer_source(name)}
        vals_by_dim = {d: [] for d in QUALITY_DIMS}
        for col_idx, dim in enumerate(header):
            if dim in vals_by_dim:
                vals_by_dim[dim].append(as_binary(r.iloc[col_idx]))
        for dim, vals in vals_by_dim.items():
            clean = [v for v in vals if not pd.isna(v)]
            row[f"n_{dim}_ratings"] = len(clean)
            row[f"{dim}_majority"] = int(np.mean(clean) >= 0.5) if clean else np.nan
            row[f"{dim}_mean"] = float(np.mean(clean)) if clean else np.nan
            row[f"{dim}_all_agree"] = int(len(set(clean)) == 1) if clean else 0
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, {"rows": int(len(df)), "duplicate_image_names": int(df["image_name"].duplicated().sum()) if len(df) else 0}

def read_quality_workbook(path: Path) -> Tuple[pd.DataFrame, Dict]:
    xl = pd.ExcelFile(path)
    rows = []
    sheet_meta = {}
    for sheet in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        header_row = None
        for i in range(min(8, len(raw))):
            vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            if "image name" in vals and ("illumination" in vals or "gold standard" in vals) and "overall" in vals:
                header_row = i
                break
        if header_row is None:
            sheet_meta[sheet] = {"status": "no_header", "rows": int(len(raw))}
            continue
        header = [norm_dim(v) for v in raw.iloc[header_row].tolist()]
        local_rows = []
        for _, r in raw.iloc[header_row + 1:].iterrows():
            if "image name" not in header:
                continue
            name = r.iloc[header.index("image name")]
            if pd.isna(name):
                continue
            name = str(name).strip()
            row = {"image_name": name, "sheet": sheet, "source_from_official_name": infer_source(name)}
            for dim in QUALITY_DIMS:
                if dim in header:
                    row[f"{dim}_official"] = as_binary(r.iloc[header.index(dim)])
            if "gold standard" in header:
                row["illumination_official"] = as_binary(r.iloc[header.index("gold standard")])
            local_rows.append(row)
        sheet_df = pd.DataFrame(local_rows)
        counts = sheet_df["source_from_official_name"].value_counts().to_dict() if len(sheet_df) else {}
        sheet_meta[sheet] = {"status": "parsed", "rows": int(len(sheet_df)), "source_counts": {str(k): int(v) for k, v in counts.items()}}
        rows.extend(local_rows)
    df = pd.DataFrame(rows)
    return df, {"sheets": sheet_meta, "rows": int(len(df)), "duplicate_image_names": int(df["image_name"].duplicated().sum()) if len(df) else 0}

def agreement_summary(df: pd.DataFrame) -> Dict:
    out = {}
    for dim in QUALITY_DIMS:
        col = f"{dim}_majority"
        agree_col = f"{dim}_all_agree"
        valid = df[col].notna() if col in df else pd.Series([], dtype=bool)
        out[dim] = {"n_valid": int(valid.sum()), "positive_rate": float(df.loc[valid, col].mean()) if valid.any() else None, "all_annotators_agree_rate": float(df.loc[valid, agree_col].mean()) if valid.any() else None}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mshf-root", default="MSHF_dataset_2.0/MSHF dataset 2.0")
    ap.add_argument("--out-dir", default="research_mshf/outputs")
    args = ap.parse_args()
    root = Path(args.mshf_root)
    out = Path(args.out_dir)
    manifest_dir = out / "manifests"
    audit_dir = out / "audit"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    images = scan_images(root / "Original")
    individual, individual_meta = read_individual_labels(root / "Individual_scores.xlsx.xlsx")
    official, official_meta = read_quality_workbook(root / "MSHF_quality_scores.xlsx")
    manifest = images.merge(individual, on="image_name", how="outer")
    if len(official):
        manifest = manifest.merge(official.drop_duplicates("image_name", keep="first"), on="image_name", how="left")
    manifest["has_image"] = manifest["relative_path"].notna()
    label_cols = [f"{d}_majority" for d in QUALITY_DIMS]
    manifest["has_all_majority_labels"] = manifest[label_cols].notna().all(axis=1)
    manifest["source"] = manifest["source_from_name"].fillna(manifest.get("source_from_label", "unknown"))
    mismatch = {}
    for dim in QUALITY_DIMS:
        mcol, ocol = f"{dim}_majority", f"{dim}_official"
        if mcol in manifest and ocol in manifest:
            valid = manifest[mcol].notna() & manifest[ocol].notna()
            mismatch[dim] = {"n_comparable": int(valid.sum()), "n_disagree": int((manifest.loc[valid, mcol] != manifest.loc[valid, ocol]).sum()) if valid.any() else 0}
    report = {
        "mshf_root": str(root),
        "image_files_original": int(len(images)),
        "unique_image_names_original": int(images["image_name"].nunique()) if len(images) else 0,
        "image_source_counts": {str(k): int(v) for k, v in images["source_from_name"].value_counts().items()} if len(images) else {},
        "individual_labels": individual_meta,
        "official_labels": official_meta,
        "majority_label_summary": agreement_summary(individual),
        "manifest_rows": int(len(manifest)),
        "manifest_has_image_rows": int(manifest["has_image"].sum()),
        "manifest_has_all_majority_labels_rows": int(manifest["has_all_majority_labels"].sum()),
        "rows_missing_image": int((~manifest["has_image"]).sum()),
        "rows_missing_majority_labels": int((~manifest["has_all_majority_labels"]).sum()),
        "majority_vs_official_mismatch": mismatch,
        "critical_warnings": [],
    }
    if report["rows_missing_image"]:
        report["critical_warnings"].append("Some labeled rows do not have images under Original.")
    if report["rows_missing_majority_labels"]:
        report["critical_warnings"].append("Some image rows lack complete individual-majority labels.")
    for sheet, sm in official_meta.get("sheets", {}).items():
        if sm.get("status") == "parsed":
            counts = sm.get("source_counts", {})
            if not (sheet in counts and len(counts) == 1):
                report["critical_warnings"].append(f"Official workbook sheet {sheet} has mixed/unexpected source names: {counts}")
    manifest_path = manifest_dir / "mshf_manifest.csv"
    audit_path = audit_dir / "mshf_audit.json"
    manifest.to_csv(manifest_path, index=False)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"WROTE {manifest_path}")
    print(f"WROTE {audit_path}")

if __name__ == "__main__":
    main()
