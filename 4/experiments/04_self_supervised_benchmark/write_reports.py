from bootstrap import configuration
from src.self_supervised.reports import write
from src.self_supervised.utils import artifact_inventory

if __name__=="__main__":
    write(configuration())
    artifact_inventory()

