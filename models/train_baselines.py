"""Member 3: reproducible Caltech rolling one-hour-ahead baselines.

Target y[t] counts starts in [t, t+1h). Issued at t, after [t-1h,t)
has completed; latest observed count is y[t-1]. Forecast horizon is one
hour to target interval end, NOT a fixed-origin forecast of the whole test.
Source clock is preserved (CSV has no timezone); no local-time claim made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scores(y, prediction):
    error = np.asarray(prediction) - np.asarray(y)
    return {'MAE': float(np.abs(error).mean()),
            'RMSE': float(np.sqrt(np.mean(error ** 2)))}


def zero_runs(frame):
    rows = []
    for site, g in frame.groupby('site'):
        g = g.sort_values('hour').copy()
        zeros = g.session_starts.eq(0)
        for _, run in g.groupby(zeros.ne(zeros.shift()).cumsum()):
            if run.session_starts.eq(0).all() and len(run) >= 24:
                rows.append(dict(site=site, start_hour=run.hour.iloc[0],
                                 end_hour=run.hour.iloc[-1], hours=len(run)))
    return pd.DataFrame(rows, columns=['site', 'start_hour', 'end_hour', 'hours'])


def audit_inputs(root):
    paths = [root/'member1_outputs'/name for name in
             ['session_start_times.csv', 'data_summary.csv', 'unparsed_files.csv']]
    paths += [root/'member2_outputs'/name for name in
              ['hourly_counts.csv', 'hourly_summary.csv', 'long_zero_runs.csv']]
    for path in paths:
        require(path.is_file(), f'Missing required input: {path}')
    tables = {p.name: pd.read_csv(p) for p in paths}
    audit = {'files': {p.name: {'rows': len(tables[p.name]),
             'columns': list(tables[p.name].columns),
             'dtypes_read': tables[p.name].dtypes.astype(str).to_dict(),
             'missing': tables[p.name].isna().sum().to_dict(),
             'duplicate_rows': int(tables[p.name].duplicated().sum()),
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}}
    h = tables['hourly_counts.csv'].copy()
    require(set(['hour', 'site', 'session_starts']).issubset(h.columns),
            'hourly_counts.csv requires hour, site, session_starts')
    require(not h.isna().any().any(), 'Missing hourly values; no implicit imputation')
    h['hour'] = pd.to_datetime(h.hour, errors='raise')
    h['session_starts'] = pd.to_numeric(h.session_starts, errors='raise')
    require(np.isfinite(h.session_starts).all() and (h.session_starts >= 0).all()
            and (h.session_starts % 1 == 0).all(), 'Counts must be nonnegative integers')
    require(not h.duplicated(['site', 'hour']).any(), 'Duplicate site/hour keys')
    require((h.hour == h.hour.dt.floor('h')).all(), 'Hours must align to hour boundaries')
    h = h.sort_values(['site', 'hour']).reset_index(drop=True)
    s = tables['session_start_times.csv'].copy()
    require(not s.isna().any().any(), 'Missing session data')
    s['session_start'] = pd.to_datetime(s.session_start, errors='raise')
    require(not s.file_path.duplicated().any(), 'Duplicate source file paths')
    s['hour'] = s.session_start.dt.floor('h')
    recount = s.groupby(['site', 'hour']).size()
    hourly = h.set_index(['site', 'hour']).session_starts
    require(len(recount.index.difference(hourly.index)) == 0, 'Sessions outside hourly coverage')
    require(hourly.eq(recount.reindex(hourly.index, fill_value=0)).all(),
            'Hourly counts do not match Member 1 sessions')
    audit['hourly_recount_mismatches'] = 0
    audit['sites'] = {}
    for site, g in h.groupby('site'):
        require(g.hour.diff().dropna().eq(pd.Timedelta(hours=1)).all(),
                f'{site}: missing hours; do not silently fill with zeros')
        sg = s[s.site.eq(site)]
        meta = tables['hourly_summary.csv'].query('site == @site')
        require(len(meta) == 1, f'{site}: invalid hourly summary')
        meta = meta.iloc[0]
        actual = dict(hours_in_series=len(g), zero_hours=int(g.session_starts.eq(0).sum()),
                      nonzero_hours=int(g.session_starts.ne(0).sum()),
                      total_session_starts=int(g.session_starts.sum()),
                      max_starts_in_one_hour=int(g.session_starts.max()))
        for field, value in actual.items():
            require(int(meta[field]) == value, f'{site}: summary mismatch {field}')
        require(pd.Timestamp(meta.first_hour) == g.hour.min() and
                pd.Timestamp(meta.last_hour) == g.hour.max(), 'Hourly coverage mismatch')
        summary = tables['data_summary.csv'].query('site == @site')
        require(int(summary.parsed_sessions.sum()) == len(sg), 'Session summary mismatch')
        require(int(summary.file_count.sum()) == len(sg) + int(summary.failed_files.sum()),
                'File count mismatch')
        require(pd.to_datetime(summary.first_start).min() == sg.session_start.min() and
                pd.to_datetime(summary.last_start).max() == sg.session_start.max(),
                'Session timestamp range mismatch')
        actual.update(first_hour=str(g.hour.min()), last_hour=str(g.hour.max()),
                      zero_fraction=float(g.session_starts.eq(0).mean()))
        audit['sites'][site] = actual
    require(int(tables['data_summary.csv'].failed_files.sum()) ==
            len(tables['unparsed_files.csv']), 'Unparsed-file summary mismatch')
    runs = zero_runs(h)
    supplied = tables['long_zero_runs.csv'].copy()
    for col in ['start_hour', 'end_hour']:
        supplied[col] = pd.to_datetime(supplied[col], errors='raise')
    require(supplied.is_zero.eq(True).all(), 'Invalid long_zero_runs is_zero flags')
    cols = ['site', 'start_hour', 'end_hour', 'hours']
    pd.testing.assert_frame_equal(runs[cols].sort_values(cols).reset_index(drop=True),
                                  supplied[cols].sort_values(cols).reset_index(drop=True),
                                  check_dtype=False)
    for site, g in runs.groupby('site'):
        audit['sites'][site].update(long_zero_run_count=len(g),
                                    long_zero_run_hours=int(g.hours.sum()),
                                    longest_zero_run_hours=int(g.hours.max()))
    audit['warnings'] = [
        'Zero counts measure absence of records, not verified absence of demand; no coverage log supplied.',
        'All zeros and extreme counts retained; no outcome-dependent test filtering.',
        'Source timestamps are timezone-naive; preserve source clock. Raw spot checks show +00:00.',
        'Filename/session-start proxy is not independently verified vehicle arrival time.',
        'Boundary hours may be partially observed; demand regime changes limit generalization.',
        'JPL audited only; no cross-site experiment in this stage.'
    ]
    return h, runs, audit


def make_features(series):
    """Index is target interval START. Every demand feature is strictly lagged."""
    idx = series.index
    x = pd.DataFrame(index=idx)
    # Fixed category definitions require no full-data fitting.
    for hour in range(1, 24):
        x[f'hour_{hour}'] = (idx.hour == hour).astype(float)
    for day in range(1, 7):
        x[f'weekday_{day}'] = (idx.dayofweek == day).astype(float)
    x['weekend'] = (idx.dayofweek >= 5).astype(float)
    for lag in [1, 24, 168]:
        x[f'lag_{lag}'] = series.shift(lag)
    x['rolling_mean_24'] = series.shift(1).rolling(24, min_periods=24).mean()
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Folder containing member1_outputs and member2_outputs')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--site', default='Caltech')
    args = parser.parse_args()
    h, runs, audit = audit_inputs(args.data_root)
    require(args.site in h.site.unique(), f'Unknown site {args.site}')
    series = h[h.site.eq(args.site)].set_index('hour').session_starts
    x = make_features(series).dropna()
    y = series.loc[x.index]
    n = len(x)
    require(n >= 1000, 'Too few usable hours for the specified preliminary experiment')
    a, b = int(n * .70), int(n * .85)
    train, validation, test = x.iloc[:a], x.iloc[a:b], x.iloc[b:]
    require(train.index.max() < validation.index.min() < test.index.min(), 'Split overlap')
    tuning = []
    candidates = [0.1, 1.0, 10.0, 100.0, 1000.0]
    for alpha in candidates:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd'))
        model.fit(train, y.loc[train.index])
        pred = model.predict(validation)
        tuning.append(dict(alpha=alpha, **scores(y.loc[validation.index], pred)))
    selected = min(tuning, key=lambda row: (row['MAE'], row['alpha']))['alpha']
    # Freeze the training-only fit after validation selection. Do NOT refit on
    # validation; scaler mean/variance and coefficients use train only.
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected, solver='svd'))
    model.fit(train, y.loc[train.index])
    predictions = pd.DataFrame({
        'forecast_origin_last_observed_hour': test.index - pd.Timedelta(hours=1),
        'forecast_issued_at': test.index,
        'target_hour': test.index,
        'target_end': test.index + pd.Timedelta(hours=1),
        'site': args.site, 'horizon_hours': 1,
        'actual': y.loc[test.index].to_numpy(),
        'seasonal_naive': test.lag_24.to_numpy(),
        'ridge': model.predict(test),
    })
    # Retrospective quality flag, never used as a feature or selection rule.
    flag = np.zeros(len(predictions), dtype=bool)
    for run in runs[runs.site.eq(args.site)].itertuples():
        flag |= predictions.target_hour.between(run.start_hour, run.end_hour).to_numpy()
    predictions['in_long_zero_run_retrospective'] = flag
    records = []
    for name, col in [('Seasonal Naive (lag 24h)', 'seasonal_naive'), ('Ridge Regression', 'ridge')]:
        p = predictions[col]
        records.append(dict(model=name, site=args.site, horizon_hours=1,
                            **scores(predictions.actual, p),
                            test_start=str(test.index.min()), test_end=str(test.index.max()),
                            test_n=len(test), train_n=len(train), validation_n=len(validation),
                            train_start=str(train.index.min()), train_end=str(train.index.max()),
                            validation_start=str(validation.index.min()),
                            validation_end=str(validation.index.max()),
                            ridge_alpha=selected if col == 'ridge' else None,
                            negative_predictions=int(p.lt(0).sum()), min_prediction=float(p.min()),
                            max_prediction=float(p.max()), clipping='none',
                            evaluation='rolling one-hour; frozen training-only fit'))
    result = pd.DataFrame(records)
    diagnostics = []
    for subset, mask in [('all', np.ones(len(predictions), dtype=bool)),
                         ('observed_zero', predictions.actual.eq(0)),
                         ('observed_nonzero', predictions.actual.gt(0)),
                         ('long_zero_run_retrospective', flag),
                         ('outside_long_zero_run_retrospective', ~flag)]:
        for col in ['seasonal_naive', 'ridge']:
            sub = predictions.loc[mask]
            if len(sub):
                diagnostics.append(dict(subset=subset, model=col, n=len(sub),
                                        **scores(sub.actual, sub[col])))
    audit['experiment'] = dict(site=args.site, usable_n=n, warmup_hours=168,
        split='floor(0.70*n), floor(0.85*n); chronological; identical test targets',
        selected_alpha=selected, selection_metric='validation MAE',
        refit_validation=False, feature_names=list(x.columns),
        test_zero_fraction=float(predictions.actual.eq(0).mean()),
        test_long_zero_run_hours=int(flag.sum()),
        target_definition='y[t] in [t,t+1h); issue at t, latest completed y[t-1]',
        clock='source timezone-naive; no conversion',
        split_statistics={label: dict(n=len(frame), start=str(frame.index.min()),
            end=str(frame.index.max()), mean=float(y.loc[frame.index].mean()),
            zero_fraction=float(y.loc[frame.index].eq(0).mean()))
            for label, frame in [('train', train), ('validation', validation), ('test', test)]})
    audit['versions'] = dict(python=platform.python_version(), numpy=np.__version__,
                            pandas=pd.__version__, sklearn=sklearn.__version__,
                            matplotlib=matplotlib.__version__)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out/'model_results.csv', index=False)
    predictions.to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(tuning).to_csv(out/'validation_results.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out/'error_diagnostics.csv', index=False)
    (out/'data_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    h.assign(month=h.hour.dt.to_period('M').astype(str)).groupby(['site','month']).agg(
        hours=('session_starts','size'), total=('session_starts','sum'),
        mean=('session_starts','mean'), maximum=('session_starts','max')).to_csv(out/'monthly_quality_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, metric in zip(axes, ['MAE', 'RMSE']):
        bars = ax.bar(['Seasonal Naive', 'Ridge'], result[metric], color=['#387aab', '#dc8043'], width=.55)
        ax.bar_label(bars, fmt='%.4f', padding=4)
        ax.set_ylim(0, result[metric].max() * 1.22)
        ax.set_ylabel(f'{metric} (session starts/hour)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle(f'{args.site} | rolling 1-hour forecast | raw predictions\n'
                 f'{test.index.min():%Y-%m-%d %H:%M} to {test.index.max():%Y-%m-%d %H:%M} | n={len(test):,}', fontsize=12)
    fig.savefig(out/'model_comparison.png', dpi=180)
    plt.close(fig)
    print(result.to_string(index=False))
    print('Versions:', audit['versions'])
    print('Split statistics:', audit['experiment']['split_statistics'])


if __name__ == '__main__':
    main()
