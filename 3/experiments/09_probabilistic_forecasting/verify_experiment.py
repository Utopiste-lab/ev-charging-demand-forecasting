"""Independent sequential reproducibility assertions for Experiment 09."""

from math import ceil

import numpy as np
import pandas as pd

from stage3_context import CASE_SITES, HOUR, PROB_METRICS, PROB_TABLES, ROOT, Project, config, read, timestamp_hash
from run_rolling_validation import windows as rolling_windows
from src.feature_engineering import feature_catalog
from src.probabilistic_forecasting.calibration import release_split

KEY = ["window", "case", "horizon_hours"]
PARTS = KEY + ["partition"]
Q = {"P10": .1, "P50": .5, "P90": .9}


def close(got, expected):
    np.testing.assert_allclose(got, np.asarray(expected, dtype=float), rtol=1e-11, atol=1e-11, equal_nan=True)


def interval_inputs(data, method):
    lo, hi = ("P10", "P90") if method == "raw" else ("CQR_lower", "CQR_upper")
    y = data.actual.to_numpy()
    l, u = data[lo].to_numpy(), data[hi].to_numpy()
    return y, l, u


def score_rows(y, l, u, nominal):
    below, above = y < l, y > u
    cover = (~below & ~above).astype(float)
    width = u - l
    score = width + (2 / (1 - nominal)) * (np.where(below, l-y, 0) + np.where(above, y-u, 0))
    return np.column_stack([cover, width, score]), above, below


def verify_temporal(project, cfg, out, forecast):
    def load(name, **opts):
        return read(out / PROB_TABLES / (name + ".csv"), **opts)
    attest = load("temporal_verification").set_index(KEY + ["partition"]).sort_index()
    adjustments = load("calibration_adjustments").set_index(KEY)
    selection = load("selected_hyperparameters").set_index(KEY).sort_index()
    tuning = load("tuning_predictions", parse_dates=["target_hour"]).groupby(KEY + ["model", "candidate"], sort=False)
    windows = dict(rolling_windows())
    seen = 0
    for key, data in forecast.groupby(KEY, sort=False):
        w, case, h = key
        src, tgt = CASE_SITES[case]
        cal = data.loc[data.partition.eq("calibration")]
        test = data.loc[data.partition.eq("test")]
        assert cal.observation_site.eq(src).all() and test.observation_site.eq(tgt).all()
        calendar = project.windows if w == "official" else windows[w]
        fc = case if src == tgt else src + " matched"
        src_data, _ = project.dataset(fc, src, h, calendar)
        dst_data, _ = project.dataset(case, tgt, h, calendar)
        days = cfg["calibration_days"]["official" if w == "official" else "rolling"]
        keys, start = release_split(src_data, calendar, days)
        for role, ids in keys.items():
            row = attest.loc[(*key, role)]
            assert row.n == len(ids) and row.timestamp_sha256 == timestamp_hash(ids)
            assert row.fit_or_observation_site == (tgt if role == "test" else src)
            assert pd.Timestamp(row.calendar_calibration_start) == start and row.future_perturbation == "passed"
            assert pd.Timestamp(row.last_label_available) == ids[-1] + HOUR
            assert pd.Timestamp(row.first_issue) == ids[0] - (h-1)*HOUR
        assert keys["final_fit"][-1] + HOUR <= cal.forecast_issued_at.min()
        assert keys["calibration"][-1] + HOUR <= test.forecast_issued_at.min()
        assert keys["validation"][-1] == start - h*HOUR
        assert keys["calibration"][-1] == calendar["test"][0] - h*HOUR
        for role, frame, ds, site in [("calibration", cal, src_data, src), ("test", test, dst_data, tgt)]:
            ids = pd.DatetimeIndex(frame.target_hour)
            assert ids.equals(keys[role])
            assert ids.equals(project.keys if role == "test" and w == "official" else pd.date_range(ids[0], ids[-1], freq=HOUR))
            np.testing.assert_array_equal(frame.actual, ds.y.loc[ids])
            np.testing.assert_array_equal(frame.forecast_issued_at, ids + (1-h)*HOUR)
            s = ds.y.loc[ids - project.period*HOUR]
            np.testing.assert_array_equal(frame["Seasonal Naive"], s)
            rows = project.site_rows(case, site).set_index("timestamp").session_start_count
            for name, kind, a, b in feature_catalog(project.settings, h, project.period):
                if kind == "history":
                    origin = ids - h*HOUR
                    assert b <= 0 and ((origin+b*HOUR+HOUR) <= pd.DatetimeIndex(frame.forecast_issued_at)).all()
                    lookback = np.column_stack([rows.loc[origin+i*HOUR] for i in range(a, b+1)])
                    close(ds.X.loc[ids, name], lookback.mean(axis=1))
        sorted_score = np.sort(np.maximum(cal.P10.to_numpy() - cal.actual, cal.actual - cal.P90.to_numpy()))
        rank = ceil((len(cal)+1)*cfg["coverage_nominal"])
        qhat = sorted_score[rank-1]
        check = adjustments.loc[key]
        assert check.rank_1_based == rank and check.n_calibration == len(cal)
        assert check.calibration_sha256 == timestamp_hash(keys["calibration"]) and check.frozen_before_first_test_issue
        close(check.adjustment, qhat)
        close(cal.nonconformity, np.maximum(cal.P10-cal.actual, cal.actual-cal.P90))
        close(data.CQR_lower, np.clip(data.P10 - qhat, 0, None))
        close(data.CQR_upper_before_repair, data.P90 + qhat)
        close(data.CQR_upper, np.where(data.CQR_upper_before_repair < data.CQR_lower, data.CQR_lower, data.CQR_upper_before_repair))
        np.testing.assert_array_equal(data.CQR_empty_repair, data.CQR_lower > data.CQR_upper_before_repair)
        for label, candidates in selection.loc[key].groupby("model", sort=False):
            for row in candidates.itertuples(index=False):
                values = tuning.get_group((*key, label, row.candidate))
                assert pd.DatetimeIndex(values.target_hour).equals(keys["validation"])
                np.testing.assert_array_equal(values.actual, src_data.y.loc[keys["validation"]])
                e = values.actual - values.prediction_raw
                error = abs(e) if label not in Q else np.where(e >= 0, Q[label]*e, (Q[label]-1)*e)
                close(row.validation_loss, np.mean(error))
            winner = candidates.sort_values(["validation_loss", "candidate"]).iloc[0]
            assert candidates.selected.sum() == 1 and winner.selected
        seen += 1
    assert seen == (44 if cfg["cross_site"] else 32)


def verify_scores(out, cfg, forecast):
    root = out / PROB_METRICS
    roles = forecast.groupby(PARTS, sort=False)
    nominal = cfg["coverage_nominal"]
    for row in read(root / "interval_metrics.csv").itertuples(index=False):
        data = roles.get_group((row.window, row.case, row.horizon_hours, row.partition))
        mask = {"all": np.ones(len(data), dtype=bool), "zero": data.actual.eq(0), "positive": data.actual.gt(0)}[row.regime]
        assert mask.sum() == row.n and np.isclose(np.mean(mask), row.fraction)
        if row.n:
            y, lo, hi = interval_inputs(data.loc[mask], row.method)
            scores, above, below = score_rows(y, lo, hi, nominal)
            ci, wi, si = scores.mean(axis=0)
            close([row.coverage,row.width,row.interval_score,row.coverage_minus_nominal,row.above_upper,row.below_lower],
                  [ci,wi,si,ci-nominal,above.mean(),below.mean()])
    for row in read(root / "point_metrics.csv").itertuples(index=False):
        data = roles.get_group((row.window,row.case,row.horizon_hours,row.partition))
        f, y = data[row.model].to_numpy(), data.actual.to_numpy(dtype=float)
        error = f-y
        sst, sse = np.sum((y-y.mean())**2), error @ error
        assert len(data)==row.n and ((f<0).sum()==row.negative_predictions)
        close([row.MAE,row.RMSE,row.R2,row.bias], [abs(error).mean(), (sse/len(error))**.5, 1-sse/sst if sst else np.nan,error.mean()])
        hit = (y>0) & (f>0)
        close(row.occurrence_recall, hit.sum()/(y>0).sum() if (y>0).sum() else np.nan)
        for name, mask in (("zero", y==0),("positive",y>0)):
            abs_e, sq_e = abs(error[mask]), error[mask]**2
            expected = [mask.sum(),mask.mean(),abs_e.mean() if len(abs_e) else np.nan, np.sqrt(sq_e.mean()) if len(sq_e) else np.nan]
            close([getattr(row,name+"_"+m) for m in ("n","proportion","MAE","RMSE")], expected)
    for row in read(root / "quantile_metrics.csv").itertuples(index=False):
        data = roles.get_group((row.window,row.case,row.horizon_hours,row.partition))
        for suffix, recorded in (("", row.pinball),("_raw",row.pinball_raw)):
            e = data.actual - data[row.quantile + suffix]
            close(recorded, np.where(e > 0, row.q*e, (row.q-1)*e).mean())
        close([row.empirical_strict_below, row.empirical_at_or_below],
              [(data.actual < data[row.quantile]).mean(),(data.actual <= data[row.quantile]).mean()])
    for row in read(root / "quantile_constraint_summary.csv").itertuples(index=False):
        data = roles.get_group((row.window,row.case,row.horizon_hours,row.partition))
        close([row.n,row.crossing_raw_n,row.crossing_raw_rate,row.crossing_after_n,row.negative_raw_rate,row.cqr_empty_repairs],
              [len(data),data.crossing_raw.sum(),data.crossing_raw.mean(),data.crossing_after.sum(),
               data.negative_raw.mean(),data.CQR_empty_repair.sum()])
    bootstrap = read(root / "interval_block_bootstrap.csv")
    for key, rows in bootstrap.groupby(PARTS + ["block_hours"], sort=False):
        *role, length = key
        data = roles.get_group(tuple(role))
        n = len(data)
        losses = np.column_stack([score_rows(*interval_inputs(data, m), nominal)[0] for m in ["raw","CQR"]])
        reps = cfg["bootstrap"]["replicates"]
        quot, remainder = divmod(n, length)
        draws = np.random.default_rng(cfg["seed"]+length).integers(0,n-length+1,size=(reps,quot+bool(remainder)))
        matrix = np.lib.stride_tricks.sliding_window_view(losses,length,axis=0).sum(axis=-1)[draws[:,:quot]].sum(axis=1)
        if remainder:
            matrix += np.lib.stride_tricks.sliding_window_view(losses,remainder,axis=0).sum(axis=-1)[draws[:,-1]]
        for row in rows.itertuples(index=False):
            i = ("coverage","width","interval_score").index(row.metric)
            j = i+3
            indices = {"raw": (i,), "CQR": (j,), "CQR minus raw": (j,i)}[row.statistic]
            values = matrix[:,indices[0]]
            actual = losses[:,indices[0]]
            if len(indices)==2:
                values = values-matrix[:,indices[1]]
                actual = actual-losses[:,indices[1]]
            estimate, avg = actual.mean(), values.mean()/n
            l, u = np.quantile(values/n,[.025,.975])
            assert row.replicates == reps and row.n == n
            close([row.estimate,row.bootstrap_mean,row.ci_low,row.ci_high], [estimate,avg,l,u])


def verify_saved(project, cfg, out):
    from src.probabilistic_forecasting.conformal import correction, interval
    # All real E09 adjustments can equal zero for a discrete count target.
    # Verify nonzero logic separately with fixed small samples.
    assert correction([-2, -1, 0, 1, 2, 4], .8) == (4., 6)
    assert correction([-4, -3, -2, -2, -2, -1], .8) == (-1., 6)
    lo, hi, raw_hi = interval([0, 2], [0, 5], -1)
    np.testing.assert_array_equal((lo, hi, raw_hi), ([1, 3], [1, 4], [-1, 4]))
    np.testing.assert_array_equal(interval([1], [2], 4), ([0], [6], [6]))
    predictions = read(out / PROB_TABLES / "probabilistic_predictions.csv", parse_dates=["target_hour","forecast_issued_at"])
    assert not predictions.duplicated(PARTS+["target_hour"]).any()
    raw = predictions[["P10_raw","P50_raw","P90_raw"]].to_numpy()
    ordered = predictions[["P10","P50","P90"]].to_numpy()
    close(ordered, np.sort(np.clip(raw,0,None),axis=1))
    np.testing.assert_array_equal(predictions.crossing_raw, (raw[:,:-1] > raw[:,1:]).any(axis=1))
    np.testing.assert_array_equal(predictions.negative_raw, (raw<0).any(axis=1))
    assert not predictions.crossing_after.any() and (predictions.CQR_upper >= predictions.CQR_lower).all()
    close(predictions["Point GB pre-calibration"], np.maximum(0,predictions["Point GB pre-calibration raw"]))
    assert predictions["Constant Zero"].eq(0).all()
    verify_temporal(project,cfg,out,predictions)
    verify_scores(out,cfg,predictions)
    print(f"Independently verified {len(predictions):,} calibration/test rows, signed CQR, alignment, split hashes,"
          " quantile ordering, selection loss and block-bootstrap metrics.",flush=True)


if __name__ == "__main__":
    verify_saved(Project(), config("config/probabilistic_forecasting.yaml"),ROOT)
