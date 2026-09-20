"""Download pinned original data and reproduce the study's prepared numerical series."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def numeric(frame):
    return frame.drop(columns=['date','timestamp','rv1','rv2'],errors='ignore').select_dtypes('number')


def prepare(name, spec, raw, output):
    source = pd.read_csv(raw)
    if name.startswith('bdg2'):
        # Fix both IDs and order, including historical tie choices in missingness ranking.
        data = source[spec['columns']].ffill().bfill()
        dates = source.iloc[:,0]
        if 'snapshot_adjustments' in spec:
            # Archived Rat predates the recoverable preparation script. Preserve its
            # 78 differing cells explicitly; do not invent a historical cleaning rule.
            adjustments = pd.read_csv(ROOT/'config'/spec['snapshot_adjustments'])
            for row in adjustments.itertuples():
                if np.float32(data.at[row.row,row.meter]) != np.float32(row.filled_source_value):
                    raise ValueError('Snapshot adjustment source does not match')
                data.at[row.row,row.meter] = row.archived_value
    else:
        data = numeric(source)
        dates = source.iloc[:,0]
    if list(data.columns) != spec['columns'] or len(data) != spec['rows']:
        raise ValueError(f'{name}: unexpected rows or ordered columns')
    values = np.ascontiguousarray(data.values,dtype='<f4')
    if not np.isfinite(values).all(): raise ValueError(f'{name}: unfilled or nonfinite values')
    actual = hashlib.sha256(values.tobytes()).hexdigest()
    if actual != spec['float32_sha256']:
        raise ValueError(f'{name}: numeric hash differs from the evaluated series: {actual}')
    result = data.copy(); result.insert(0,'date',dates)
    result.to_csv(output,index=False)
    reloaded = np.ascontiguousarray(numeric(pd.read_csv(output)).values,dtype='<f4')
    if hashlib.sha256(reloaded.tobytes()).hexdigest() != actual:
        raise ValueError('CSV round-trip changed model inputs')
    return dict(dataset=name,rows=len(data),channels=len(data.columns),float32_sha256=actual,
                source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),url=spec['url'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'data/prepared')
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw')
    args=parser.parse_args();specs=json.loads((ROOT/'config/datasets.json').read_text())
    names=list(specs) if args.datasets=='all' else args.datasets.split(',')
    if not set(names)<=set(specs):parser.error('Unknown dataset')
    args.output.mkdir(parents=True,exist_ok=True);args.raw_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in names:
        spec=specs[name];filename='electricity_cleaned.csv' if name.startswith('bdg2') else name+'.csv'
        raw=args.raw_dir/filename
        is_pointer = raw.exists() and raw.stat().st_size < 1024 and raw.read_bytes().startswith(b'version https://git-lfs.github.com/spec/')
        if not raw.exists() or is_pointer:
            request=urllib.request.Request(spec['url'],headers={'User-Agent':'endpoint-reproduction'})
            temporary=raw.with_suffix('.download')
            with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as handle:
                while block:=response.read(1024*1024):handle.write(block)
            temporary.replace(raw)
        record=prepare(name,spec,raw,args.output/(name+'.csv'));records.append(record)
        print(name,'numeric hash verified',record['rows'],record['channels'],flush=True)
    (args.output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
