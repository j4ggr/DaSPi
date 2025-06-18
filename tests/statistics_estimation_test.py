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

from daspi import SpecLimits
from daspi import Specification
from daspi.statistics.estimation import *
from daspi import load_dataset


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


class TestMeasurementUncertainty:
    """Test the MeasurementUncertainty class."""
    
    def test_initialization_with_standard(self) -> None:
        """Test initialization with standard uncertainty."""
        uncertainty = MeasurementUncertainty(standard=0.05)
        assert uncertainty.standard == 0.05
        assert uncertainty.expanded == 0.1  # Default k=2
        assert uncertainty.error_limit == approx(0.086602, rel=1e-5)  # √3 * standard

    def test_initialization_with_error_limit(self) -> None:
        """Test initialization with error limit."""
        uncertainty = MeasurementUncertainty(error_limit=0.1)
        assert uncertainty.standard == approx(0.057735, rel=1e-5)  # √3 ≈ 1.732
        assert uncertainty.expanded == approx(0.11547, rel=1e-5)  # Default k=2
        assert uncertainty.k == 2

    def test_initialization_with_expanded(self) -> None:
        """Test initialization with expanded uncertainty."""
        uncertainty = MeasurementUncertainty(expanded=0.2)
        assert uncertainty.standard == approx(0.1, rel=1e-5) # Default k=2
        assert uncertainty.expanded == 0.2
        assert uncertainty.error_limit == approx(0.173205, rel=1e-5)  # √3 * standard

    def test_initialization_with_invalid_distribution(self) -> None:
        """Test initialization with invalid distribution."""
        with pytest.raises(AssertionError):
            MeasurementUncertainty(distribution='invalid') # type: ignore

    def test_initialization_with_confidence_level(self) -> None:
        """Test initialization with confidence level."""
        uncertainty = MeasurementUncertainty(standard=0.05, confidence_level=0.99)
        assert uncertainty.confidence_level == 0.99
    
    def test_with_zero_standard(self) -> None:
        """Test initialization with zero standard uncertainty."""
        u_0 = MeasurementUncertainty(standard=0.0)
        u_1 = MeasurementUncertainty(standard=0.1)
        assert u_0.standard == 0
        assert u_0.expanded == 0
        assert u_0.error_limit == 0

        u_tot = u_0.combine_with(u_1)
        assert u_tot.standard == u_1.standard
        assert u_tot.expanded == u_1.expanded
        assert u_tot.error_limit == u_1.error_limit

    def test_relative_uncertainty(self) -> None:
        """Test calculation of relative uncertainty."""
        uncertainty = MeasurementUncertainty(standard=0.1)
        relative = uncertainty.relative(1.0)
        assert relative == 10.0

    def test_combine_with_rss(self) -> None:
        """Test combining uncertainties using root sum of squares."""
        u1 = MeasurementUncertainty(standard=0.1)
        u2 = MeasurementUncertainty(standard=0.2)
        combined = u1.combine_with(u2)
        assert combined.standard == approx(0.223607, rel=1e-5)  # √(0.1² + 0.2²)

    def test_combine_with_linear(self) -> None:
        """Test combining uncertainties using linear addition."""
        u1 = MeasurementUncertainty(standard=0.1)
        u2 = MeasurementUncertainty(standard=0.2)
        combined = u1.combine_with(u2, method='linear')
        assert combined.standard == approx(0.3)

    def test_multiplication(self) -> None:
        """Test multiplication of uncertainty."""
        uncertainty = MeasurementUncertainty(standard=0.1)
        result = uncertainty * 2
        assert result.standard == 0.2

    def test_equality(self) -> None:
        """Test equality comparison."""
        u1 = MeasurementUncertainty(standard=0.1)
        u2 = MeasurementUncertainty(standard=0.1)
        u3 = MeasurementUncertainty(standard=0.2)
        assert u1 == u2
        assert u1 != u3
        assert u2 <= u1
        assert u1 >= u2
        assert u3 > u1
        assert u2 < u3
        assert not u3 < u2
        assert not u1 > u3

    def test_summary(self) -> None:
        """Test summary method."""
        uncertainty = MeasurementUncertainty(standard=0.1)
        summary = uncertainty.summary()
        assert summary['standard'] == 0.1
        assert summary['expanded'] == 0.2  # Default k=2
        assert summary['error_limit'] == approx(0.173205, rel=1e-5)  # √3 * standard
        assert summary['distribution'] == 'rectangular'


class TestDistributionEstimator:

    # source data contains 8 decimal places
    rel_curve: float = 1e-7
    estimate: DistributionEstimator = DistributionEstimator(df_dist10['rayleigh'])
    
    @pytest.fixture
    def sample_estimator(self) -> DistributionEstimator:
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        return DistributionEstimator(data)
    
    def test_data_filtered(self) -> None:
        N = 10
        N_nan = 2
        data = np.concatenate((np.random.randn(N-2), N_nan*[np.nan], [.1, -.1]))
        estimate = DistributionEstimator(data)
        assert len(estimate.samples) == N + N_nan
        assert estimate._filtered.empty
        assert len(estimate.filtered) == N
        assert not estimate._filtered.empty
        assert not any(np.isnan(estimate.filtered))

    def test_excess(self) -> None:
        rel = self.rel_curve

        size = 10
        for dist in df_dist10.columns:
            estimate = DistributionEstimator(df_dist10[dist])
            excess = estimate.excess
            assert excess == approx(df_valid10[dist]['excess'], rel=rel)
        
        size = 25
        for dist in df_dist10.columns:
            estimate = DistributionEstimator(df_dist25[dist])
            excess = estimate.excess
            assert excess == approx(df_valid25[dist]['excess'], rel=rel)

    def test_skew(self) -> None:
        rel = self.rel_curve

        size = 10
        for dist in df_dist10.columns:
            estimate = DistributionEstimator(df_dist10[dist])
            skew = estimate.skew
            assert skew == approx(df_valid10[dist]['skew'], rel=rel)
        
        size = 25
        for dist in df_dist10.columns:
            estimate = DistributionEstimator(df_dist25[dist])
            skew = estimate.skew
            assert skew == approx(df_valid25[dist]['skew'], rel=rel)

    def test_follows_norm_curve(self) -> None:
        estimate = DistributionEstimator(df_dist25['norm'])
        assert estimate.follows_norm_curve()
        
        estimate = DistributionEstimator(df_dist25['chi2'])
        assert not estimate.follows_norm_curve()
        
        estimate = DistributionEstimator(df_dist25['foldnorm'])
        assert not estimate.follows_norm_curve()
        
        estimate = DistributionEstimator(df_dist25['weibull_min'])
        assert not estimate.follows_norm_curve()
        
        estimate = DistributionEstimator(df_dist25['gamma'])
        assert not estimate.follows_norm_curve()
        
        estimate = DistributionEstimator(df_dist25['wald'])
        assert not estimate.follows_norm_curve()

        estimate = DistributionEstimator(df_dist25['expon'])
        assert not estimate.follows_norm_curve()

    def test_stable_variance(self) -> None:
        assert self.estimate.stable_variance()

        data = list(df_dist25['logistic']) + list(df_dist25['expon'])
        estimate = DistributionEstimator(data)
        assert not estimate.stable_variance(n_sections=2)

    def test_fit_distribution(self) -> None:
        estimate = DistributionEstimator(df_dist25['expon'])
        assert estimate._dist is None
        assert estimate._shape_params is None
        assert estimate._p_ks is None
       
        estimate.distribution()
        assert estimate.dist.name != 'norm'
        assert estimate.p_ks > 0.005
        assert estimate.shape_params is not None
        
        estimate = DistributionEstimator(df_dist25['norm'])
        estimate.distribution()
        assert estimate.p_ks > 0.005
        assert estimate.dist.name != 'expon'

    def test_describe_basic(self, sample_estimator: DistributionEstimator) -> None:
        result = sample_estimator.describe()
        assert 'n_samples' in result.index
        assert 'n_missing' in result.index
        assert 'dist_name' in result.index
        assert 'p_ks' in result.index
        assert 'p_ad' in result.index
        assert 'excess' in result.index
        assert 'p_excess' in result.index
        assert 'skew' in result.index
        assert 'p_skew' in result.index
        assert 'ss' in result.index
        assert 'aic' in result.index
        assert 'bic' in result.index

    def test_describe_with_exclude(self, sample_estimator: DistributionEstimator) -> None:
        result = sample_estimator.describe(exclude=('ss', 'aic', 'bic'))
        assert 'n_samples' in result.index
        assert 'n_missing' in result.index
        assert 'dist_name' in result.index
        assert 'p_ks' in result.index
        assert 'p_ad' in result.index
        assert 'excess' in result.index
        assert 'p_excess' in result.index
        assert 'skew' in result.index
        assert 'p_skew' in result.index
        assert 'ss' not in result.index
        assert 'aic' not in result.index
        assert 'bic' not in result.index

    def test_describe_with_empty_exclude(self, sample_estimator: DistributionEstimator) -> None:
        result = sample_estimator.describe(exclude=())
        attrs = sample_estimator.attrs_describe
        assert all(a in result.index for a in attrs)

    def test_describe_with_nan_values(self) -> None:
        data = [1.0, np.nan, 3.0, 4.0, np.nan]
        estimator = DistributionEstimator(data)
        result = estimator.describe()
        assert estimator.n_missing == 2
        assert estimator.n_samples == 5
        assert estimator.n_filtered == 3
        assert not any(pd.isna(result.loc[['excess', 'skew']]))
        assert len(result) == len(estimator.attrs_describe)

    def test_nan_policy_warn(self) -> None:
        data = [1.0, np.nan, 3.0, 4.0, np.nan]
        with pytest.warns(UserWarning):
            estimator = DistributionEstimator(data)
        assert estimator.nan_policy == 'omit'

        with pytest.warns(UserWarning):
            estimator = DistributionEstimator(data, nan_policy='propagate')
        assert estimator.nan_policy == 'propagate'

    def test_nan_policy_raise(self) -> None:
        data = [1.0, np.nan, 3.0, 4.0, np.nan]
        with pytest.raises(ValueError):
            DistributionEstimator(data, nan_policy='raise')


class TestLocationDispersionEstimator:

    # source data contains 8 decimal places
    rel_curve: float = 1e-7
    estimate: LocationDispersionEstimator = LocationDispersionEstimator(df_dist10['rayleigh'])
    
    @pytest.fixture
    def sample_estimator(self) -> LocationDispersionEstimator:
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        return LocationDispersionEstimator(data)
    
    def test_data_filtered(self) -> None:
        N = 10
        N_nan = 2
        data = np.concatenate((np.random.randn(N-2), N_nan*[np.nan], [.1, -.1]))
        estimate = LocationDispersionEstimator(data)
        assert len(estimate.samples) == N + N_nan
        assert estimate._filtered.empty
        assert len(estimate.filtered) == N
        assert not estimate._filtered.empty
        assert not any(np.isnan(estimate.filtered))

    def test_min(self) -> None:
        assert self.estimate._min is None
        assert self.estimate.min == approx(np.min(self.estimate.filtered))
        assert self.estimate._min is not None

    def test_max(self) -> None:
        assert self.estimate._max is None
        assert self.estimate.max == approx(np.max(self.estimate.filtered))
        assert self.estimate._max is not None

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

    def test_describe_basic(self, sample_estimator: LocationDispersionEstimator) -> None:
        result = sample_estimator.describe()
        assert 'min' in result.index
        assert 'max' in result.index
        assert 'mean' in result.index
        assert 'median' in result.index
        assert 'std' in result.index

    def test_describe_with_exclude(self, sample_estimator: LocationDispersionEstimator) -> None:
        result = sample_estimator.describe(exclude=('mean', 'median'))
        assert 'mean' not in result.index
        assert 'median' not in result.index
        assert 'min' in result.index
        assert 'max' in result.index

    def test_full_range(self) -> None:
        data = np.random.normal(size=10_000)

        estimator = LocationDispersionEstimator(data, strategy='data', agreement=1.0)
        assert estimator.agreement == float('inf')
        assert estimator._k == float('inf')
        assert estimator.lcl == estimator.min
        assert estimator.ucl == estimator.max
        
        estimator = LocationDispersionEstimator(data, strategy='data', agreement=float('inf'))
        assert estimator.agreement == float('inf')
        assert estimator._k == float('inf')
        assert estimator.lcl == estimator.min
        assert estimator.ucl == estimator.max
        
        estimator = LocationDispersionEstimator(data, strategy='norm', agreement=1.0)
        assert estimator.agreement == float('inf')
        assert estimator._k == float('inf')
        assert estimator.lcl == float('-inf')
        assert estimator.ucl == float('inf')

        estimator = LocationDispersionEstimator(data, strategy='data', agreement=1)
        assert estimator.agreement == 1
        assert estimator._k == 0.5
        assert estimator.lcl != estimator.min
        assert estimator.ucl != estimator.max


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
        loess = Loess(pd.DataFrame({'x': [1], 'y': [1]}), 'y', 'x', fit_at_init=False)
        kernels = loess.available_kernels
        assert 'tricube' in kernels
        assert 'gaussian' in kernels
        assert 'epanechnikov' in kernels

    def test_fit_predict(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x')
        assert hasattr(loess, 'smoothed')
        assert len(loess.smoothed) == len(sample_data)
        
        # Test prediction
        pred = loess.predict(5.0)
        assert isinstance(pred, np.ndarray)
        assert len(pred) == 1

    def test_fitted_line(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x')
        
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
        loess = Loess(pd.DataFrame({'x': [1], 'y': [1]}), 'y', 'x', fit_at_init=False)
        with pytest.raises(AssertionError):
            loess.fit(fraction=0.3, kernel='invalid_kernel') # type: ignore

    def test_predict_before_fit(self, sample_data: DataFrame) -> None:
        loess = Loess(sample_data, target='y', feature='x', fit_at_init=False)
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

    @pytest.fixture
    def sample_estimator(self) -> ProcessEstimator:
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        return ProcessEstimator(data, SpecLimits(upper=2))
    
    @pytest.fixture
    def estimator_norm(self) -> ProcessEstimator:
        return ProcessEstimator(df_dist25['norm'], SpecLimits(lower=0))
    
    def test_init_with_series(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'], SpecLimits())
        assert len(estimator.samples) == 100
        assert estimator._filtered.empty
        assert len(estimator.filtered) == 100

    def test_init_with_nan_values(self) -> None:
        data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        estimator = ProcessEstimator(data, SpecLimits())
        assert len(estimator.samples) == 5
        assert len(estimator.filtered) == 3
        assert list(estimator.filtered) == [1.0, 3.0, 5.0]

    def test_inheritance_methods(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'], SpecLimits())
        assert hasattr(estimator, 'mean')
        assert hasattr(estimator, 'median')
        assert hasattr(estimator, 'std')
        assert hasattr(estimator, 'skew')
        assert hasattr(estimator, 'excess')

    def test_process_specific_methods(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'], SpecLimits())
        assert hasattr(estimator, 'describe')
        assert callable(estimator.describe)

    def test_describe_output(self, sample_data: DataFrame) -> None:
        estimator = ProcessEstimator(sample_data['values'], SpecLimits())
        description = estimator.describe()
        expected_keys = (
            'lsl', 'usl', 'n_ok', 'n_nok', 'n_errors', 'cp', 'cpk', 'Z', 'Z_lt')
        assert all([key in description.index for key in expected_keys])

    def test_with_all_identical_values(self) -> None:
        data = pd.Series([1.0] * 10)
        estimator = ProcessEstimator(data, SpecLimits())
        assert estimator.std == 0
        assert np.isnan(estimator.skew)
        assert np.isnan(estimator.excess)

    def test_with_extreme_values(self) -> None:
        data = pd.Series([1e10, 1e-10, 1e5, 1e-5])
        estimator = ProcessEstimator(data, SpecLimits())
        assert not np.isnan(estimator.mean)
        assert not np.isnan(estimator.std)
        assert not np.isnan(estimator.skew)
        assert not np.isnan(estimator.excess)

    def test_describe_basic(self, sample_estimator: ProcessEstimator) -> None:
        result = sample_estimator.describe()
        assert 'n_ok' in result.index
        assert 'n_nok' in result.index
        assert 'n_errors' in result.index
        assert 'ok' in result.index
        assert 'nok' in result.index
        assert 'min' in result.index
        assert 'max' in result.index
        assert 'mean' in result.index
        assert 'median' in result.index
        assert 'std' in result.index
        assert 'dist_name' in result.index
        assert 'lcl' in result.index
        assert 'ucl' in result.index
        assert 'strategy' in result.index
        assert 'Z' in result.index
        assert 'Z_lt' in result.index

    def test_describe_with_exclude(self, sample_estimator: ProcessEstimator) -> None:
        result = sample_estimator.describe(exclude=('dist_name', 'strategy'))
        assert 'dist_name' not in result.index
        assert 'strategy' not in result.index
        assert 'lcl' in result.index
        assert 'ucl' in result.index

    def test_describe_with_empty_exclude(self, sample_estimator: ProcessEstimator) -> None:
        result = sample_estimator.describe(exclude=())
        expected_attrs = sample_estimator.attrs_describe
        assert all(attr in result.index for attr in expected_attrs)

    def test_describe_with_nan_values(self) -> None:
        data = [1.0, np.nan, 3.0, 4.0, np.nan]
        estimator = ProcessEstimator(data, SpecLimits(lower=1.5, upper=3.5))
        result = estimator.describe()
        assert not np.isnan(result.loc['mean'][0])
        assert not np.isnan(result.loc['std'][0])
        assert len(result) == len(estimator.attrs_describe)
    
    def test_n_nok(self, estimator_norm: ProcessEstimator) -> None:
        assert estimator_norm._n_nok is None
        result = estimator_norm.n_nok
        assert isinstance(result, int)
        assert isinstance(estimator_norm._n_nok, int)
        assert 0 < estimator_norm._n_nok < estimator_norm.n_samples

    def test_nok_norm(self, estimator_norm: ProcessEstimator) -> None:
        assert estimator_norm._nok_norm is None
        result = estimator_norm.nok_norm
        assert isinstance(result, str)
        assert isinstance(estimator_norm._nok_norm, float)
        assert 0 < estimator_norm._nok_norm < 1
        assert f'{100 * estimator_norm._nok_norm:.2f}' in result
        estimator_norm.nok
        assert estimator_norm._nok_norm > estimator_norm._nok

    def test_nok_fit(self, estimator_norm: ProcessEstimator) -> None:
        assert estimator_norm._nok_fit is None
        result = estimator_norm.nok_fit
        assert isinstance(result, str)
        assert isinstance(estimator_norm._nok_fit, float)
        assert 0 < estimator_norm._nok_fit < 1
        assert f'{100 * estimator_norm._nok_fit:.2f}' in result
        estimator_norm.nok
        estimator_norm.nok_norm
        assert estimator_norm._nok_fit < estimator_norm._nok
        assert estimator_norm._nok_fit < estimator_norm._nok_norm


class TestGageEstimator:
    df = load_dataset('grnr_layer_thickness')

    @pytest.fixture
    def estimator_gage(self) -> GageEstimator:
        estimator = GageEstimator(
            self.df['result_gage'].dropna(),
            reference=0.101,
            u_cal=0.0002,
            tolerance=Specification(limits=(0.085, 0.115)),
            resolution=0.001)
        return estimator

    def test_specification_values(self, estimator_gage: GageEstimator) -> None:
        assert estimator_gage.nominal == pytest.approx(0.101)
        assert estimator_gage.tolerance == pytest.approx(0.03)
        assert estimator_gage.lower == pytest.approx(0.0980)
        assert estimator_gage.upper == pytest.approx(0.1040)
        assert estimator_gage.tolerance_adj == pytest.approx(0.006)

    def test_measured_values(self, estimator_gage: GageEstimator) -> None:
        assert estimator_gage.min == pytest.approx(0.0990)
        assert estimator_gage.max == pytest.approx(0.1020)
        assert estimator_gage.R == pytest.approx(0.0030)
        assert estimator_gage.n_samples == 50
    
    def test_statistic_values(self, estimator_gage: GageEstimator) -> None:
        assert estimator_gage.mean == pytest.approx(0.10066000, abs=1e-8)
        assert estimator_gage.std == pytest.approx(0.00068839, abs=1e-8)
        assert estimator_gage.lcl == pytest.approx(0.09928322, abs=1e-8)
        assert estimator_gage.ucl == pytest.approx(0.10203677, abs=1e-8)
        assert estimator_gage.bias == pytest.approx(-0.00034000, abs=1e-8)
    
    def test_capable_values(self, estimator_gage: GageEstimator) -> None:
        assert estimator_gage.cg == pytest.approx(2.17900482, abs=1e-8)
        assert estimator_gage.cgk == pytest.approx(1.93205094, abs=1e-8)
        assert estimator_gage.resolution_ratio == pytest.approx(1/30)
        assert estimator_gage.T_min_cg == pytest.approx(0.01831111, abs=1e-8)
        assert estimator_gage.T_min_cgk == pytest.approx(0.02171111, abs=1e-8)
        assert estimator_gage.T_min_res == pytest.approx(0.02000000, abs=1e-8)

    def test_check(self, estimator_gage: GageEstimator) -> None:
        assert estimator_gage.check() == {
            'U_cal': True,
            'resolution': True,
            'cg': True,
            'cgk': True,}

    def test_estimate_resolution(self) -> None:
        specification=SpecLimits(lower=20.15, upper=20.45)
        estimator = GageEstimator(
            self.df['result_gage'],
            reference=0.101,
            u_cal=0.0002,
            tolerance=specification,
            resolution=None)
        assert estimator.resolution == 0.001

        estimator = GageEstimator(
            [1.0, 1.01, 1.002, 1.0003],
            reference=0.101,
            u_cal=0.0002,
            tolerance=specification,
            resolution=None)
        assert estimator.resolution == 0.0001

        estimator = GageEstimator(
            [1, 20, 300, 4000],
            reference=0.101,
            u_cal=0.0002,
            tolerance=specification,
            resolution=None)
        assert estimator.resolution == 1

    def test_uncertainties(self) -> None:
        pass