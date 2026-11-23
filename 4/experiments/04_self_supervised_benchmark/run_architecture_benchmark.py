import argparse

from bootstrap import configuration
from src.self_supervised.benchmark import run_stage, smoke

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--smoke-only",action="store_true")
    opts=args.parse_args()
    cfg=configuration()
    if opts.smoke_only:
        smoke(cfg)
    else:
        run_stage("architecture",cfg)

