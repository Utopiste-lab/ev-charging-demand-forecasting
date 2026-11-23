from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.self_supervised.utils import FIGURE, LOG, METRIC, MODEL, TABLE, configuration

for output in (FIGURE, LOG, METRIC, MODEL, TABLE):
    output.mkdir(exist_ok=True,parents=True)

