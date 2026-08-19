"""Preprocessing for the FuzzyPharma PLGA drug-release study.

Typical use::

    from fuzzypharma import build_dataset

    ds = build_dataset()
    ds.model_table   # 321 formulations: static features + targets + fold
    ds.curves        # 321 x 50 resampled release curves on normalized time

Feature representations are built per fold so that scaler statistics and fuzzy
knots never see held-out data::

    from fuzzypharma.features import Representation, build_xy
    from fuzzypharma.folds import iter_folds

    for fold, train, valid in iter_folds(ds.model_table):
        Xtr, Xva, ytr, yva, rep = build_xy(
            ds.model_table, "Release_24h", Representation("hybrid"), train, valid
        )
"""

from .pipeline import Dataset, build_dataset

__all__ = ["Dataset", "build_dataset"]
__version__ = "0.1.0"
