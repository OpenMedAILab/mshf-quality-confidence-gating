#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd


def build_split(base: Path, split_name: str, csv_name: str):
    csv_path = base / split_name / csv_name
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        image_id = str(r['image_id'])
        patient_id = int(r['patient_id'])
        eye = 'left' if '_l' in image_id else 'right' if '_r' in image_id else 'unknown'
        dr = r['left_eye_DR_Level'] if eye == 'left' else r['right_eye_DR_Level'] if eye == 'right' else r['patient_DR_Level']
        img_path = base / split_name / 'Images' / str(patient_id) / f'{image_id}.jpg'
        rows.append({
            'split': 'train' if 'training' in split_name else 'validation',
            'patient_id': patient_id,
            'image_id': image_id,
            'image_name': f'{image_id}.jpg',
            'eye': eye,
            'view': image_id[-1] if image_id[-1].isdigit() else '',
            'image_path': str(img_path),
            'exists': img_path.exists(),
            'dr_grade': dr,
            'referable_dr': int(float(dr) >= 2) if pd.notna(dr) else None,
            'patient_dr_grade': r.get('patient_DR_Level'),
            'deepdrid_overall_quality': r.get('Overall quality'),
            'deepdrid_clarity': r.get('Clarity'),
            'deepdrid_field_definition': r.get('Field definition'),
            'deepdrid_artifact': r.get('Artifact'),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='research_mshf/data/external/DeepDRiD/extracted/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images')
    ap.add_argument('--out-csv', default='research_mshf/outputs/manifests/deepdrid_regular_manifest.csv')
    ap.add_argument('--report-json', default='research_mshf/outputs/audit/deepdrid_regular_report.json')
    a = ap.parse_args()
    base = Path(a.root)
    train = build_split(base, 'regular-fundus-training', 'regular-fundus-training.csv')
    val = build_split(base, 'regular-fundus-validation', 'regular-fundus-validation.csv')
    manifest = pd.concat([train, val], ignore_index=True)
    out = Path(a.out_csv); out.parent.mkdir(parents=True, exist_ok=True); manifest.to_csv(out, index=False)
    report = {
        'root': str(base),
        'rows': int(len(manifest)),
        'missing_images': int((~manifest['exists']).sum()),
        'split_counts': {str(k): int(v) for k, v in manifest['split'].value_counts().items()},
        'patient_counts': {str(k): int(v) for k, v in manifest.groupby('split')['patient_id'].nunique().items()},
        'grade_counts': {str(k): int(v) for k, v in manifest['dr_grade'].value_counts(dropna=False).sort_index().items()},
        'referable_counts': {str(k): int(v) for k, v in manifest['referable_dr'].value_counts(dropna=False).items()},
        'out_csv': str(out),
    }
    rpath = Path(a.report_json); rpath.parent.mkdir(parents=True, exist_ok=True); rpath.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True)); print('WROTE', out); print('WROTE', rpath)

if __name__ == '__main__':
    main()
