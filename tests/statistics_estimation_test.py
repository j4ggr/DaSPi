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


class TestLoess:
    @pytest.fixture
    def sample_data(self) -> DataFrame:
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 1, 100)
        df = pd.DataFrame({'x': x, 'y': y})
        return df

    def test_init(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x')
        assert isinstance(loess.source, pd.DataFrame)
        assert loess.target == 'y'
        assert loess.feature == 'x'
        assert len(loess.x) == len(sample_data)

    def test_empty_data(self):
        empty_df = pd.DataFrame({'x': [], 'y': []})
        with pytest.raises(AssertionError, match='No data after removing missing values'):
            Loess(empty_df, target='y', feature='x')

    def test_available_kernels(self) -> None:
        loess = Loess(pd.DataFrame({'x': [1], 'y': [1]}), 'y', 'x')
        kernels = loess.available_kernels
        assert 'tricube' in kernels
        assert 'gaussian' in kernels
        assert 'epanechnikov' in kernels

    def test_fit_predict(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x')
        fitted = loess.fit()
        assert fitted is loess
        assert hasattr(loess, 'smoothed')
        assert len(loess.smoothed) == len(sample_data)
        
        # Test prediction
        pred = loess.predict(5.0)
        assert isinstance(pred, np.ndarray)
        assert len(pred) == 1

    def test_fitted_line(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x')
        loess.fit()
        
        # Without confidence intervals
        seq, pred = loess.fitted_line(confidence_level=None, n_points=50)
        assert len(seq) == 50
        assert len(pred) == 50
        
        # With confidence intervals
        seq, pred, lower, upper = loess.fitted_line(confidence_level=0.95, n_points=50)
        assert len(seq) == 50
        assert len(pred) == 50
        assert len(lower) == 50
        assert len(upper) == 50
        assert all(lower <= upper)

    def test_invalid_kernel(self) -> None:
        loess = Loess(pd.DataFrame({'x': [1], 'y': [1]}), 'y', 'x')
        with pytest.raises(AssertionError):
            loess.fit(fraction=0.3, kernel='invalid_kernel') # type: ignore

    def test_predict_before_fit(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x')
        with pytest.raises(AssertionError):
            loess.predict(5.0)

    def test_residuals(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x')
        loess.fit(fraction=0.3)
        residuals = loess.residuals
        assert len(residuals) == len(sample_data)
        assert isinstance(residuals, pd.Series)


# TODO: Add tests for cp and cpk
class TestProcessEstimator:

    @pytest.fixture
    def sample_data(self) -> DataFrame:
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        return pd.DataFrame({'values': data})

    def test_init_with_series(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'])
        assert len(estimator.samples) == 100
        assert estimator._filtered.empty
        assert len(estimator.filtered) == 100

    def test_init_with_nan_values(self) -> None:
        data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        estimator = ProcessEstimator(data)
        assert len(estimator.samples) == 5
        assert len(estimator.filtered) == 3
        assert list(estimator.filtered) == [1.0, 3.0, 5.0]

    def test_inheritance_methods(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'])
        assert hasattr(estimator, 'mean')
        assert hasattr(estimator, 'median')
        assert hasattr(estimator, 'std')
        assert hasattr(estimator, 'skew')
        assert hasattr(estimator, 'excess')

    def test_process_specific_methods(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'])
        assert hasattr(estimator, 'describe')
        assert callable(estimator.describe)

    def test_describe_output(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'])
        description = estimator.describe()
        expected_keys = (
            'lsl', 'usl', 'n_ok', 'n_nok', 'n_errors', 'cp', 'cpk', 'Z', 'Z_lt')
        assert all([key in description.index for key in expected_keys])

    def test_with_all_identical_values(self) -> None:
        data = pd.Series([1.0] * 10)
        estimator = ProcessEstimator(data)
        assert estimator.std == 0
        assert np.isnan(estimator.skew)
        assert np.isnan(estimator.excess)

    def test_with_extreme_values(self) -> None:
        data = pd.Series([1e10, 1e-10, 1e5, 1e-5])
        estimator = ProcessEstimator(data)
        assert not np.isnan(estimator.mean)
        assert not np.isnan(estimator.std)
        assert not np.isnan(estimator.skew)
        assert not np.isnan(estimator.excess)
