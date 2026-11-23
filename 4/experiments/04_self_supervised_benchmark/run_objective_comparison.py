from bootstrap import configuration
from src.self_supervised.benchmark import run_stage

if __name__=="__main__":
    run_stage("objective",configuration())

