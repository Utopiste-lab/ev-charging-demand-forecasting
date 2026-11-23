import json

from bootstrap import configuration
import pandas as pd

from src.self_supervised.presentation import figures
from src.self_supervised.utils import METRIC, TABLE

if __name__=="__main__":
    chosen=json.loads((TABLE/"selected_architecture.json").read_text("utf-8"))["architecture"]
    figures(configuration(),pd.read_csv(METRIC/"all_scores.csv"),pd.read_csv(METRIC/"training_resources.csv"),chosen)

