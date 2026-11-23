from pathlib import Path
import os
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.full_dataset.common import LOG
from src.full_dataset.quality import generate_tables, figures

if __name__ == "__main__":
    os.environ["MPLCONFIGDIR"]=str(LOG/"matplotlib")
    figures(generate_tables())

