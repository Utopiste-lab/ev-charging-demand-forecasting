"""Generate model-independent leakage attestations for the chronological meta-dataset."""

from audit_previous_results import timestamp_hash
from src.adaptive_hybrid.temporal_oof import build_oof


def generate(project, fit, data, ledger, h, block_count, prefix=None):
    meta, x, manifest, prefix = build_oof(fit, data, ledger, project.cfg, project.advanced,
                                          timestamp_hash, block_count, prefix)
    assert meta.index.isin(fit.train.y.index).all() and meta.index.intersection(fit.validation.y.index).empty
    assert meta.actual.equals(fit.train.y.loc[meta.index])
    assert all(r["last_fit_label_available"] <= r["first_issue"]
               and r["last_prefix_label_available"] <= r["first_issue"] and r["purged"] == h - 1
               for r in manifest)
    return meta, x, manifest, prefix
