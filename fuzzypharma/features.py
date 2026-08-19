"""Crisp, fuzzy-only, and hybrid feature representations.

These are the three arms of the controlled comparison in the proposal. Each is
a fit/transform object, so the scaler statistics and the fuzzy knots are both
learned inside a training fold and only applied to the held-out fold.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import numpy as np

from .config import FORMULATION_METHODS, METHOD_COL, STATIC_FEATURES
from .fuzzy import FuzzyFeaturizer

CRISP = "crisp"
FUZZY = "fuzzy"
HYBRID = "hybrid"
REPRESENTATIONS = (CRISP, FUZZY, HYBRID)


def encode_formulation_method(
    values: pd.Series, categories: list[str] = FORMULATION_METHODS
) -> pd.DataFrame:
    """One-hot encode the emulsion method against a declared vocabulary.

    The benchmark treats formulation method as a nominal variable and one-hot
    encodes it. Encoding against a fixed, declared category list (rather than
    fitting on whatever a fold happens to contain) keeps the feature width equal
    across folds and involves no target information.

    This matters concretely: ``S/W/O/W`` occurs in exactly one formulation in the
    entire dataset, so a per-fold encoder would emit a column in some folds and
    not others. Its column is simply all-zero in any training fold that does not
    contain it, which is the honest representation of a method the model has
    never seen.
    """
    known = pd.Categorical(values, categories=categories)
    if known.isna().any():
        unexpected = sorted(set(values[pd.isna(known)]))
        raise ValueError(f"Unknown formulation method(s): {unexpected}")

    encoded = pd.get_dummies(known, prefix=METHOD_COL, prefix_sep=" = ")
    encoded.index = values.index
    return encoded.astype(float)


class Representation(BaseEstimator, TransformerMixin):
    """Build one of the three feature representations.

    Parameters
    ----------
    kind:
        ``"crisp"``   original numeric features only (the baseline)
        ``"fuzzy"``   membership degrees only
        ``"hybrid"``  both, concatenated
    scale_crisp:
        Standardize the crisp block. Membership degrees are left on their
        native [0, 1] scale either way, since rescaling them would distort the
        linguistic meaning of a degree.
    include_method:
        Append the one-hot formulation method. It is a nominal input in all
        three arms -- fuzzification applies to the numeric features, so the
        fuzzy-only arm still needs the categorical input to be comparable.
    features:
        Numeric input columns. Defaults to the ten static features.
    fuzzy_kwargs:
        Forwarded to :class:`~fuzzypharma.fuzzy.FuzzyFeaturizer`.
    """

    def __init__(
        self,
        kind: str = CRISP,
        scale_crisp: bool = True,
        include_method: bool = True,
        features: list[str] | None = None,
        **fuzzy_kwargs,
    ):
        self.kind = kind
        self.scale_crisp = scale_crisp
        self.include_method = include_method
        self.features = features
        self.fuzzy_kwargs = fuzzy_kwargs

    def fit(self, X: pd.DataFrame, y=None):
        if self.kind not in REPRESENTATIONS:
            raise ValueError(f"kind must be one of {REPRESENTATIONS}, got {self.kind!r}")

        self.features_ = list(self.features) if self.features is not None else list(STATIC_FEATURES)
        missing = sorted(set(self.features_) - set(X.columns))
        if missing:
            raise ValueError(f"Input columns absent: {missing}")

        self.use_method_ = self.include_method and METHOD_COL in X.columns
        block = X[self.features_]

        self.scaler_ = None
        if self.kind in (CRISP, HYBRID) and self.scale_crisp:
            self.scaler_ = StandardScaler().fit(block)

        self.fuzzifier_ = None
        if self.kind in (FUZZY, HYBRID):
            self.fuzzifier_ = FuzzyFeaturizer(
                features=self.features_, **self.fuzzy_kwargs
            ).fit(block)

        self.feature_names_out_ = list(self.transform(X.head(1)).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "features_"):
            raise RuntimeError("Representation must be fitted before transform.")

        block = X[self.features_]
        parts = []

        if self.kind in (CRISP, HYBRID):
            if self.scaler_ is not None:
                crisp = pd.DataFrame(
                    self.scaler_.transform(block),
                    columns=self.features_,
                    index=X.index,
                )
            else:
                crisp = block.copy()
            parts.append(crisp)

        if self.kind in (FUZZY, HYBRID):
            parts.append(self.fuzzifier_.transform(block))

        if self.use_method_:
            parts.append(encode_formulation_method(X[METHOD_COL]))

        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_out_)


def build_xy(
    model_table: pd.DataFrame,
    target: str,
    representation: Representation,
    train_positions,
    valid_positions,
):
    """Fit a representation on the training rows and apply it to both sides.

    This is the single place experiments should go through: it makes fitting on
    the training fold and applying to the validation fold the path of least
    resistance, rather than something each script has to remember.

    Returns ``(X_train, X_valid, y_train, y_valid, representation)``.
    """
    train = model_table.iloc[train_positions]
    valid = model_table.iloc[valid_positions]

    representation = representation.fit(train)
    return (
        representation.transform(train),
        representation.transform(valid),
        train[target],
        valid[target],
        representation,
    )
