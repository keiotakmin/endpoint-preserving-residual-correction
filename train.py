"""Recreate frozen base forecasts or paired learned-adapter experiments from prepared CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from backbones import build_model,load_csv,prep
from base_training import warm_and_select,warmup_model,WARM_GRID
from joint_residual import windows
from small_joint import warm,train as train_joint
from learned_residual import predict as adapter_predict


def infer(model,data,origins):
    model.eval();out=[]
    with torch.inference_mode():
        for j in range(0,len(origins),32):
            out.append(model(torch.stack([data[t-96:t] for t in origins[j:j+32]])).cpu().numpy())
    return np.concatenate(out).astype(float)


def completed(path):
    if not path.exists(): return False
    if not all((path/name).is_file() for name in ('forecasts.npz','checkpoint.pt','training.json')):
        raise RuntimeError(f'Incomplete prior run: {path}; move it aside before retrying')
    print('COMPLETE, not overwriting',path,flush=True)
    return True


def save_run(directory,model,data,n,meta,training,adapter=None):
    directory.mkdir(parents=True,exist_ok=False)
    origins=np.arange(n,len(data)-24+1,24);base=infer(model,data,origins)
    target=np.stack([data[t:t+24].cpu().numpy() for t in origins]).astype(float)
    fields=dict(base=base,target=target,origins=origins)
    if adapter is not None:
        adapter.cpu().eval();fields['adapter']=adapter_predict(adapter,base,target)
    meta.update(mse=float(np.mean((base-target)**2)),mae=float(np.mean(abs(base-target))))
    np.savez_compressed(directory/'forecasts.npz',**fields,meta=json.dumps(meta))
    torch.save(dict(base={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    adapter=adapter.state_dict() if adapter is not None else None),directory/'checkpoint.pt')
    (directory/'training.json').write_text(json.dumps(dict(meta=meta,training=training),indent=2)+'\n')
    print('SAVED',directory.name,'MSE',meta['mse'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=['base','learned'],default='base')
    parser.add_argument('--datasets',default='ETTh1,ETTh2,ETTm1,ETTm2,appliances,bdg2')
    parser.add_argument('--backbones',default='dlinear,patchtst')
    parser.add_argument('--seeds',default='0,1,2')
    parser.add_argument('--phases',default='legacy,refit')
    parser.add_argument('--widths',default='2,4,64')
    parser.add_argument('--warmups',default='mae,sf')
    parser.add_argument('--selection',choices=['archived','validation-grid'],default='archived')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/prepared')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/training')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args();torch.set_num_threads(2)
    specs=json.loads((ROOT/'config/datasets.json').read_text())
    archived=json.loads((ROOT/'config/archived_bases.json').read_text())
    lookup={(r['dataset'],r['backbone'],r['seed'],r['phase']):r for r in archived}
    args.output.mkdir(parents=True,exist_ok=True)
    for name in args.datasets.split(','):
        file=args.data_dir/(name+'.csv');values=load_csv(file)
        digest=hashlib.sha256(np.ascontiguousarray(values,dtype='<f4').tobytes()).hexdigest()
        if digest!=specs[name]['float32_sha256']:raise ValueError(f'{name}: data differs from the paper; prepare_data.py first')
        data,n,c=prep(values,device=args.device);a=int(n*.8)
        for backbone in args.backbones.split(','):
            if backbone not in ('dlinear','patchtst'):raise ValueError(backbone)
            for seed in map(int,args.seeds.split(',')):
                common=dict(dataset=name,backbone=backbone,seed=seed,device=args.device,
                            data_float32_sha256=digest,torch_version=torch.__version__,numpy_version=np.__version__)
                if args.group=='base':
                    ref=lookup[(name,backbone,seed,'legacy')]
                    selected=None;selected_model=None
                    for phase in args.phases.split(','):
                        if phase not in ('legacy','refit'):raise ValueError(phase)
                        path=args.output/f'{phase}_{name}_{backbone}_s{seed}'
                        if completed(path):continue
                        begin=time.monotonic()
                        if selected is None:
                            grid=WARM_GRID if args.selection=='validation-grid' else [ref['warmup']]
                            selected_model,selected,val=warm_and_select(backbone,96,24,c,data,a,n,seed,warm_grid=grid)
                        if phase=='legacy':model=selected_model
                        else:
                            torch.manual_seed(seed);np.random.seed(seed)
                            model=warmup_model(backbone,96,24,c,data,n,selected,device=args.device)
                        reference=lookup.get((name,backbone,seed,phase))
                        meta=dict(common,group='base',phase=phase,warmup=selected,selection=args.selection,
                                  reference_mse=reference['reference_mse'] if reference else None,
                                  training_seconds=time.monotonic()-begin)
                        save_run(path,model,data,n,meta,dict(selected_steps=selected,validation_mse=val))
                else:
                    tr,va=windows(data,0,a),windows(data,a,n)
                    torch.manual_seed(seed);fresh=build_model(backbone,96,24,c).to(args.device)
                    for kind in args.warmups.split(','):
                        if kind not in ('mae','sf'):raise ValueError(kind)
                        warmbase,wlog=warm(fresh,tr,kind=='sf')
                        # The matched base-only run is shared by all widths at this warmup/seed.
                        for width in [0]+list(map(int,args.widths.split(','))):
                            tag='base_only' if width==0 else f'joint_r{width}'
                            path=args.output/f'{kind}_{tag}_{name}_{backbone}_s{seed}'
                            if completed(path):continue
                            begin=time.monotonic()
                            model,adapter,state,log,epoch=train_joint(warmbase,tr,va,width!=0,seed=seed,rank=width or 64)
                            meta=dict(common,group='learned',phase=tag,kind=kind,width=width,selected_epoch=epoch,
                                      training_seconds=time.monotonic()-begin)
                            save_run(path,model,data,n,meta,dict(warmup=wlog,joint=log),adapter)
        del data
        if args.device.startswith('cuda'):torch.cuda.empty_cache()


if __name__=='__main__':main()
