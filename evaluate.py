"""Evaluate recreated forecasts without changing archived numerical records."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from endpoint_review_validation import configs, evaluate, intervals, select_prefix
from channel_control import RollingControl


def blended_losses(base, target, raw):
    gate = RollingControl(base.shape[-1], 'global', 'blend', 32)
    loss = []
    for b, y, p in zip(base, target, raw):
        correction = p - b
        issued = b + gate.alpha()[0] * correction
        loss.append([[np.mean((p-y)**2), np.mean(abs(p-y))],
                     [np.mean((issued-y)**2), np.mean(abs(issued-y))]])
        gate.update(correction, y-b)
    return np.asarray(loss)


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'runs/training')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/evaluation')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/prepared')
    parser.add_argument('--sensitivity', action='store_true', help='All 21 candidates and temporal selection')
    parser.add_argument('--fourier', action='store_true', help='Seven ELF configurations on base group')
    args = parser.parse_args()
    paths = sorted(args.input.glob('*/forecasts.npz'))
    if not paths: parser.error('No forecasts.npz found; run train.py first')
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = configs() if args.sensitivity else configs()[:1]
    rows, selections = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            base, target = data['base'], data['target']
            origins = data['origins']; meta = json.loads(str(data['meta']))
            adapter = data['adapter'] if 'adapter' in data else None
        static = np.stack([np.mean((base-target)**2, axis=(1,2)),
                           np.mean(abs(base-target), axis=(1,2))], axis=-1)
        values = [evaluate(base, target, cfg) for cfg in candidates]
        losses = np.stack([v['loss'] for v in values])
        names = [cfg.name for cfg in candidates]
        sizes = np.array([[v['raw_bytes'], v['state_bytes']] for v in values])
        np.savez_compressed(args.output/('blocks_'+path.parent.name+'.npz'),
                            names=names, losses=losses, static=static, sizes=sizes,
                            alpha=np.stack([v['alpha'] for v in values]), meta=json.dumps(meta))
        all_methods = dict(zip(names, losses))
        if adapter is not None:
            all_methods['adapter'] = blended_losses(base, target, adapter)
        if args.fourier and meta['group'] == 'base':
            from backbones import load_csv, prep
            from elf_real import forecast_stream
            import hashlib
            specs = json.loads((ROOT/'config/datasets.json').read_text())
            raw = load_csv(args.data_dir/(meta['dataset']+'.csv'))
            digest = hashlib.sha256(np.ascontiguousarray(raw,dtype='<f4').tobytes()).hexdigest()
            if digest != specs[meta['dataset']]['float32_sha256']:
                raise ValueError('Fourier input differs from the evaluated series')
            series, _, _ = prep(raw, device='cpu')
            season = 144 if meta['dataset']=='appliances' else 96 if meta['dataset'].startswith('ETTm') else 24
            for label, d, q in [('full',87,10),('in3',3,10),('in5',5,10),('in9',9,10),
                                ('out4',87,4),('compact3',3,4),('compact5',5,4)]:
                prediction = forecast_stream(series.numpy(), origins, season, d, q)['predictions']
                all_methods['elf_'+label] = blended_losses(base, target, prediction)
        identity = dict(stream=path.parent.name, dataset=meta['dataset'], backbone=meta['backbone'],
                        seed=meta['seed'], group=meta['group'], phase=meta['phase'], kind=meta.get('kind',''))
        periods = [('full',0,len(base))] + intervals(len(base))
        for name, loss in all_methods.items():
            for segment, start, stop in periods:
                for p, policy in enumerate(['raw','global']):
                    mse, mae = loss[start:stop,p].mean(axis=0)
                    ref_mse, ref_mae = static[start:stop].mean(axis=0)
                    rows.append(dict(identity, candidate=name, segment=segment, policy=policy,
                                     start=start, stop=stop, mse=mse, mae=mae,
                                     static_mse=ref_mse, static_mae=ref_mae))
        if args.sensitivity:
            for segment, start, stop in intervals(len(base)):
                for pool, count in [('settings13',13),('all21',21)]:
                    ids = list(range(count))
                    chosen = select_prefix(losses, start, ids)
                    oracle = int(np.argmin(losses[:count,start:stop,1,0].mean(axis=1)))
                    selections.append(dict(identity, segment=segment, pool=pool, start=start, stop=stop,
                        selected=names[chosen], oracle=names[oracle],
                        selected_mse=float(losses[chosen,start:stop,1,0].mean()),
                        fixed_mse=float(losses[0,start:stop,1,0].mean()),
                        oracle_mse=float(losses[oracle,start:stop,1,0].mean()),
                        static_mse=float(static[start:stop,0].mean())))
        print('EVALUATED', path.parent.name, flush=True)
    write_csv(args.output/'metrics.csv', rows)
    write_csv(args.output/'selection.csv', selections)
    (args.output/'manifest.json').write_text(json.dumps(dict(
        input=str(args.input), streams=len(paths), sensitivity=args.sensitivity,
        fourier=args.fourier, records=len(rows), selection_records=len(selections)),indent=2)+'\n')


if __name__ == '__main__': main()
