import argparse
import json
import subprocess

from helpers.config import config
from helpers.log import log

if  __name__ == '__main__':



    log('Pipeline invoked...')
    cfg = config('./config.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=cfg['pipeline']['preprocess'])
    parser.add_argument('--train', type=bool, default=cfg['pipeline']['train'])
    parser.add_argument('--explain', type=bool, default=cfg['pipeline']['explain'])
    parser.add_argument('--postprocess', type=bool, default=cfg['pipeline']['postprocess'])
    args = parser.parse_args()

    print(json.dumps(cfg, indent=2))

    if args.preprocess:
        subprocess.call(['python', 'preprocess.py'])

    if args.train:
        subprocess.call(['python', 'train.py'])

    if args.explain:
        subprocess.call(['python', 'explain.py'])

    if args.postprocess:
        subprocess.call(['python', 'postprocess.py'])

    log('...pipeline done.')
