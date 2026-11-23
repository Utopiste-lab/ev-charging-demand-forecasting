"""Read-only checks for the published Stage 3 subset; no Stage 2 archive needed."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parent


def verify_manifest():
    repo = ROOT.parent
    with (repo / 'docs/artifact-index.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = (repo / row['path']).resolve()
        if not path.is_relative_to(repo.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid artifact: {row['path']}")
        if path.stat().st_size != int(row['bytes']) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f"Artifact mismatch: {row['path']}")
    print(f'PASS published manifest: {len(rows)} files')


def verify_results():
    import numpy as np
    import pandas as pd
    data = pd.read_csv(ROOT / 'results/tables/tree_models_1h_predictions.csv', float_precision='round_trip')
    metrics = pd.read_csv(ROOT / 'results/metrics/tree_models_1h_comparison.csv', float_precision='round_trip')
    expected = pd.date_range('2020-07-15', '2020-12-02 22:00:00', freq='h')
    for row in metrics.itertuples():
        for part in ['validation', 'test']:
            saved = data.loc[(data.model == row.model) & (data.prediction_variant == row.prediction_variant) & (data.partition == part)]
            if saved.empty or saved.target_hour.duplicated().any():
                raise ValueError(f'Empty or duplicate keys: {row.model}/{part}')
            if part == 'test':
                np.testing.assert_array_equal(pd.to_datetime(saved.target_hour), expected)
                assert len(saved) == row.n_test == 3383
            errors = saved.prediction.to_numpy() - saved.actual.to_numpy()
            mae = math.fsum(abs(float(x)) for x in errors) / len(errors)
            rmse = math.sqrt(math.fsum(float(x) ** 2 for x in errors) / len(errors))
            for name, value in [('MAE', mae), ('RMSE', rmse)]:
                if not math.isclose(value, getattr(row, f'{part}_{name}'), rel_tol=2e-11, abs_tol=2e-11):
                    raise ValueError(f'Score mismatch: {row.model}/{part}/{name}')
    print(f'PASS one-hour saved predictions: {len(metrics)} variants, validation/test MAE and RMSE, 3383 test keys')


def verify_features():
    import yaml
    from src.forecasting_dataset import load_hourly, build_dataset
    from src.temporal_split import calendar_windows, common_view, split_dataset
    sys.path.insert(0, str(ROOT / 'experiments/02_feature_validation'))
    from test_feature_validation import test_a_future_perturbation, test_b_feature_availability, test_c_temporal_split, test_d_target_leakage
    cfg = yaml.safe_load((ROOT / 'config/baseline_1h.yaml').read_text('utf-8'))
    hourly, _ = load_hourly(ROOT / 'data/processed/hourly_counts_v1.csv', ROOT / 'data/processed/data_provenance.json')
    common, _, _, partial = common_view(hourly)
    windows = calendar_windows(cfg['split'])
    features = cfg['features']
    period = cfg['models']['seasonal_naive']['period_hours']
    plans = 0
    for scope, frame in [('same_site', hourly), ('cross_site', common)]:
        for _, rows in frame.groupby('site'):
            counts = rows.set_index('timestamp').session_start_count.sort_index()
            for policy in ['exclude_partial', 'retain_sensitivity']:
                for horizon in [1, 3, 6, 24]:
                    dataset = build_dataset(rows, features, horizon, period, policy=policy, shared_partial_hours=partial if scope == 'cross_site' else None)
                    ledger = split_dataset(dataset, windows)
                    test_a_future_perturbation(counts, dataset, ledger, features, period)
                    test_b_feature_availability(counts, dataset, ledger, features, period)
                    test_c_temporal_split(dataset, ledger, windows, policy)
                    test_d_target_leakage(counts, dataset, ledger, features, period)
                    plans += 1
    print(f'PASS feature causality, label release and boundaries: {plans} plans; historical Stage 2 replay excluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', action='store_true', help='Also run two-site/four-horizon causality checks')
    args = parser.parse_args()
    verify_manifest()
    verify_results()
    if args.features:
        verify_features()
