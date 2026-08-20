#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd


def audit(df, risk_col, coverage):
    n=len(df); k=max(1,int(n*coverage)); accepted=set(df.sort_values(risk_col).head(k).image_id.astype(str))
    rows=[]
    for grade,g in df.groupby('dr_grade'):
        total=len(g); acc=sum(g.image_id.astype(str).isin(accepted)); rows.append({'strategy':risk_col,'dr_grade':grade,'n':int(total),'accepted':int(acc),'coverage':float(acc/total) if total else None})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--risks-csv', default='research_mshf/outputs/error_gate/deepdrid_validation_pilot/gate_input_with_risks.csv')
    ap.add_argument('--manifest', default='research_mshf/outputs/manifests/deepdrid_regular_manifest.csv')
    ap.add_argument('--coverage', type=float, default=0.8)
    ap.add_argument('--out-csv', default='research_mshf/outputs/error_gate/deepdrid_validation_pilot/severity_coverage.csv')
    ap.add_argument('--out-json', default='research_mshf/outputs/error_gate/deepdrid_validation_pilot/severity_coverage_summary.json')
    a=ap.parse_args()
    risks=pd.read_csv(a.risks_csv); man=pd.read_csv(a.manifest)
    df=risks.merge(man[['image_id','patient_id','eye','dr_grade','referable_dr','deepdrid_overall_quality']],on='image_id',how='left')
    strategies=[c for c in ['entropy_risk','quality_mean_risk','joint_risk'] if c in df.columns]
    rows=[]
    for s in strategies: rows.extend(audit(df,s,a.coverage))
    out=Path(a.out_csv); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    summary={'n':int(len(df)),'coverage':a.coverage,'strategies':strategies,'grade_counts':{str(k):int(v) for k,v in df.dr_grade.value_counts(dropna=False).sort_index().items()},'out_csv':str(out)}
    Path(a.out_json).write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(summary,indent=2,sort_keys=True)); print('WROTE',out)
if __name__=='__main__': main()
