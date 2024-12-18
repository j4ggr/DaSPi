import sys
import pytest

import numpy as np
import pandas as pd

from typing import Any
from typing import Dict
from typing import Literal
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


class TestLowess:
    @pytest.fixture
    def sample_data(self) -> DataFrame:
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 1, 100)
        df = pd.DataFrame({'x': x, 'y': y})
        return df

    def test_init(self, sample_data: DataFrame) -> None:
        lowess = Lowess(sample_data, target='y', feature='x')
        assert isinstance(lowess.source, pd.DataFrame)
        assert lowess.target == 'y'
        assert lowess.feature == 'x'
        assert len(lowess.x) == len(sample_data)

    def test_empty_data(self):
        empty_df = pd.DataFrame({'x': [], 'y': []})
        with pytest.raises(AssertionError, match='No data after removing missing values'):
            Lowess(empty_df, target='y', feature='x')

    def test_available_kernels(self) -> None:
        lowess = Lowess(pd.DataFrame({'x': [1], 'y': [1]}), 'y', 'x')
        kernels = lowess.available_kernels
        assert 'tricube' in kernels
        assert 'gaussian' in kernels
        assert 'epanechnikov' in kernels

    def test_fit_predict(self, sample_data: DataFrame) -> None:
        lowess = Lowess(sample_data, target='y', feature='x')
        fitted = lowess.fit(fraction=0.3)
        assert fitted is lowess
        assert hasattr(lowess, 'smoothed')
        assert len(lowess.smoothed) == len(sample_data)
        
        # Test prediction
        pred = lowess.predict(5.0)
        assert isinstance(pred, np.ndarray)
        assert len(pred) == 1

    def test_predict_sequence(self, sample_data: DataFrame) -> None:
        lowess = Lowess(sample_data, target='y', feature='x')
        lowess.fit(fraction=0.3)
        
        # Without confidence intervals
        seq, pred = lowess.predict_sequence(confidence_level=None, n_points=50)
        assert len(seq) == 50
        assert len(pred) == 50
        
        # With confidence intervals
        seq, pred, lower, upper = lowess.predict_sequence(confidence_level=0.95, n_points=50)
        assert len(seq) == 50
        assert len(pred) == 50
        assert len(lower) == 50
        assert len(upper) == 50
        assert all(lower <= upper)

    def test_invalid_kernel(self) -> None:
        lowess = Lowess(pd.DataFrame({'x': [1], 'y': [1]}), 'y', 'x')
        with pytest.raises(AssertionError):
            lowess.fit(fraction=0.3, kernel='invalid_kernel') # type: ignore

    def test_predict_before_fit(self, sample_data: DataFrame) -> None:
        lowess = Lowess(sample_data, target='y', feature='x')
        with pytest.raises(AssertionError):
            lowess.predict(5.0)

    def test_residuals(self, sample_data: DataFrame) -> None:
        lowess = Lowess(sample_data, target='y', feature='x')
        lowess.fit(fraction=0.3)
        residuals = lowess.residuals
        assert len(residuals) == len(sample_data)
        assert isinstance(residuals, pd.Series)
