import sys
import pytest

import numpy as np
import pandas as pd

from typing import Any
from typing import Dict
from pytest import approx
from pathlib import Path
from pandas.core.frame import DataFrame

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi.statistics.estimation import *

source = Path(__file__).parent/'data'
KW_READ: Dict[str, Any] = dict(sep=';', index_col=0)

df_dist10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', skiprows=1, nrows=10, **KW_READ)
df_valid10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', skiprows=14, **KW_READ)
df_dist25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', skiprows=1, nrows=25, **KW_READ)
df_valid25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', skiprows=29, **KW_READ)


class TestEstimator:

    # source data contains 8 decimal places
    rel_curve: float = 1e-7
    estimate: Estimator = Estimator(df_dist10['rayleigh'])

    def test_data_filtered(self) -> None:
        N = 10
        N_nan = 2
        data = np.concatenate((np.random.randn(N-2), N_nan*[np.nan], [.1, -.1]))
        estimate = Estimator(data)
        assert len(estimate.samples) == N + N_nan
        assert estimate._filtered.empty
        assert len(estimate.filtered) == N
        assert not estimate._filtered.empty
        assert not any(np.isnan(estimate.filtered))

    def test_mean(self) -> None:
        assert self.estimate._mean is None
        assert self.estimate.mean == approx(np.mean(self.estimate.filtered))
        assert self.estimate._mean is not None

    def test_median(self) -> None:
        assert self.estimate._median is None
        assert self.estimate.median == approx(np.median(self.estimate.filtered))
        assert self.estimate._median is not None

    def test_std(self) -> None:
        assert self.estimate._std is None
         # do not remove ddof=1, numpy uses ddof=0 as default!
        assert self.estimate.std == approx(np.std(self.estimate.filtered, ddof=1))
        assert self.estimate._std is not None

    def test_excess(self) -> None:
        rel = self.rel_curve

        size = 10
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist10[dist])
            excess = estimate.excess
            assert excess == approx(df_valid10[dist]['excess'], rel=rel)
        
        size = 25
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist25[dist])
            excess = estimate.excess
            assert excess == approx(df_valid25[dist]['excess'], rel=rel)

    def test_skew(self) -> None:
        rel = self.rel_curve

        size = 10
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist10[dist])
            skew = estimate.skew
            assert skew == approx(df_valid10[dist]['skew'], rel=rel)
        
        size = 25
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist25[dist])
            skew = estimate.skew
            assert skew == approx(df_valid25[dist]['skew'], rel=rel)

    def test_follows_norm_curve(self) -> None:
        estimate = Estimator(df_dist25['norm'])
        assert estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['chi2'])
        assert not estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['foldnorm'])
        assert not estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['weibull_min'])
        assert not estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['gamma'])
        assert not estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['wald'])
        assert not estimate.follows_norm_curve()

        estimate = Estimator(df_dist25['expon'])
        assert not estimate.follows_norm_curve()

    def test_stable_variance(self) -> None:
        assert self.estimate.stable_variance()

        data = list(df_dist25['logistic']) + list(df_dist25['expon'])
        estimate = Estimator(data)
        assert not estimate.stable_variance(n_sections=2)

    def test_fit_distribution(self) -> None:
        estimate = Estimator(df_dist25['expon'])
        assert estimate._dist is None
        assert estimate._dist_params is None
        assert estimate._p_dist is None
       
        estimate.distribution()
        assert estimate.dist.name != 'norm'
        assert estimate.p_dist > 0.005
        assert estimate.dist_params is not None
        
        estimate = Estimator(df_dist25['norm'])
        estimate.distribution()
        assert estimate.p_dist > 0.005
        assert estimate.dist.name != 'expon'