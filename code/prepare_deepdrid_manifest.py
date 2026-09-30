#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import pandas as pd

IMG_EXT={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
GRADE_HINTS=['grade','dr_grade','DR_grade','retinopathy grade','Retinopathy grade','level','diagnosis','label']
ID_HINTS=['image','image_name','filename','file','id','Image name','Image','fundus']

def norm(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())

def find_images(root):
    rows=[]
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            rows.append({'image_name':p.name,'stem':p.stem,'image_path':str(p),'relative_path':str(p.relative_to(root))})
    return pd.DataFrame(rows)

def read_tables(root):
    tables=[]
    for p in root.rglob('*'):
        if not p.is_file(): continue
        try:
            if p.suffix.lower()=='.csv':
                tables.append((p,pd.read_csv(p)))
            elif p.suffix.lower() in {'.xlsx','.xls'}:
                xl=pd.ExcelFile(p)
                for s in xl.sheet_names:
                    tables.append((Path(str(p)+'::'+s),pd.read_excel(p,sheet_name=s)))
        except Exception:
            pass
    return tables

def choose_columns(df):
    cols=list(df.columns); ncols={norm(c):c for c in cols}
    id_col=None; grade_col=None
    for h in ID_HINTS:
        nh=norm(h)
        for nc,c in ncols.items():
            if nh in nc or nc in nh:
                id_col=c; break
        if id_col is not None: break
    for h in GRADE_HINTS:
        nh=norm(h)
        for nc,c in ncols.items():
            if nh in nc or nc in nh:
                grade_col=c; break
        if grade_col is not None: break
    return id_col,grade_col

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--deepdrid-root', default='research_mshf/data/external/DeepDRiD')
    ap.add_argument('--out-csv', default='research_mshf/outputs/manifests/deepdrid_manifest.csv')
    ap.add_argument('--report-json', default='research_mshf/outputs/audit/deepdrid_prepare_report.json')
    a=ap.parse_args(); root=Path(a.deepdrid_root)
    report={'root':str(root),'exists':root.exists(),'status':'not_started'}
    if not root.exists():
        report['status']='missing_root'
        Path(a.report_json).parent.mkdir(parents=True,exist_ok=True); Path(a.report_json).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2)); return
    imgs=find_images(root); tables=read_tables(root)
    report.update({'image_count':int(len(imgs)),'table_count':len(tables),'tables':[str(p) for p,_ in tables[:50]]})
    best=None
    for path,df in tables:
        id_col,grade_col=choose_columns(df)
        if id_col is None or grade_col is None: continue
        tmp=df[[id_col,grade_col]].dropna().copy(); tmp.columns=['label_image_id','dr_grade']
        tmp['label_stem']=tmp.label_image_id.astype(str).map(lambda x: Path(x).stem)
        merged=imgs.merge(tmp,left_on='stem',right_on='label_stem',how='inner')
        if best is None or len(merged)>len(best[2]): best=(str(path),{'id_col':id_col,'grade_col':grade_col},merged)
    if best is None or len(best[2])==0:
        report['status']='no_matching_label_table'
    else:
        label_path,cols,manifest=best
        manifest['referable_dr']=(pd.to_numeric(manifest.dr_grade,errors='coerce')>=2).astype('Int64')
        out=Path(a.out_csv); out.parent.mkdir(parents=True,exist_ok=True); manifest.to_csv(out,index=False)
        report.update({'status':'ok','selected_label_table':label_path,'selected_columns':cols,'matched_images':int(len(manifest)),'out_csv':str(out),'grade_counts':{str(k):int(v) for k,v in manifest.dr_grade.value_counts(dropna=False).items()}})
    rpath=Path(a.report_json); rpath.parent.mkdir(parents=True,exist_ok=True); rpath.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True)); print('WROTE',rpath)
if __name__=='__main__': main()
