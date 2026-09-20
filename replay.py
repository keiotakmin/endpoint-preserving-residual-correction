"""Replay causal correction on user-supplied chronological base/target blocks."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from endpoint_review_validation import Config, evaluate

parser=argparse.ArgumentParser()
parser.add_argument('input',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise FileExistsError('Refusing to overwrite an existing output')
with np.load(args.input,allow_pickle=False) as z:
    base,target=z['base'],z['target']
if base.ndim != 3 or base.shape[1] != 24: raise ValueError('Expected [blocks,24,channels]')
result=evaluate(base,target,Config(),predictions=True)
np.savez_compressed(args.output,**result)
print('MSE/MAE rows [raw, global]:',result['loss'].mean(axis=0))
print('Retained auxiliary bytes:',result['state_bytes'])
