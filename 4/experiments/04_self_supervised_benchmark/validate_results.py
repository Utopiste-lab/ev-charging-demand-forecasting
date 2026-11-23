import argparse

from bootstrap import configuration
from src.self_supervised.verification import run

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-only",action="store_true")
    run(data_only=parser.parse_args().data_only)

