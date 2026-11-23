from bootstrap import configuration
from src.self_supervised.benchmark import assess

if __name__=="__main__":
    assess(configuration())

