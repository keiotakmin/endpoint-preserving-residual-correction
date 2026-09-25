"""Run data preparation, paired training, and chronological evaluation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='ETTh1, seed 0, both models and training groups')
    parser.add_argument('--fourier', action='store_true', help='Also evaluate all seven Fourier configurations')
    parser.add_argument('--device', default=None)
    args=parser.parse_args()
    def run(script, *options):
        subprocess.run([sys.executable, str(ROOT/script), *options], cwd=ROOT, check=True)
    run('prepare_data.py', '--datasets', 'ETTh1' if args.smoke else 'all')
    common=['--datasets','ETTh1','--seeds','0'] if args.smoke else []
    if args.device: common += ['--device',args.device]
    run('train.py', '--group','base',*common)
    run('train.py', '--group','learned',*common)
    run('evaluate.py', '--sensitivity',*(['--fourier'] if args.fourier else []))
    run('verify_training.py')


if __name__=='__main__':main()
