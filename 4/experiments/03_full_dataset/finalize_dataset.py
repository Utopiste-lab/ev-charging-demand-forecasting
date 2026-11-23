import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.full_dataset.assemble import run

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--screen-only",action="store_true")
    mode.add_argument("--reuse-screen",action="store_true")
    args=parser.parse_args()
    run(screen_only=args.screen_only,reuse_screen=args.reuse_screen)

