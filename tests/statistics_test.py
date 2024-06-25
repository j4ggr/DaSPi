import sys
import pytest

import numpy as np
import pandas as pd

from pytest import approx
from pathlib import Path
from pandas.core.series import Series
from pandas.core.frame import DataFrame

sys.path.append(Path(__file__).parent.resolve())

from daspi.statistics.confidence import *
from daspi.statistics.estimation import *
from daspi.statistics.hypothesis import *
from daspi.statistics import chunker

source = Path(__file__).parent/'data'
KW_READ = dict(sep=';', index_col=0)

df_dist10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', skiprows=1, nrows=10, **KW_READ)
df_valid10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', skiprows=14, **KW_READ)
df_dist25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', skiprows=1, nrows=25, **KW_READ)
df_valid25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', skiprows=29, **KW_READ)


class TestChunker:

    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_with_valid_input(self) -> None:
        sections = 3
        result = list(chunker(self.samples, sections))
        assert len(result) == sections
        assert np.array_equal(result[0], np.array([1, 2, 3, 4]))
        assert np.array_equal(result[1], np.array([5, 6, 7]))
        assert np.array_equal(result[2], np.array([8, 9, 10]))

    def test_with_single_section(self) -> None:
        sections = 1
        result = list(chunker(self.samples, sections))
        assert len(result) == sections
        assert np.array_equal(result[0], self.samples)

    def test_with_single_sample(self) -> None:
        sections = 2
        samples = ['foo']
        result = list(chunker(samples, sections))
        assert len(result) == sections
        assert np.array_equal(result[0], samples)
        assert result[1].size == 0

    def test_with_zero_sections(self) -> None:
        sections = 0
        with pytest.raises(AssertionError):
            list(chunker(self.samples, sections))

    def test_with_negative_sections(self) -> None:
        sections = -2
        with pytest.raises(AssertionError):
            list(chunker(self.samples, sections))

    def test_with_non_integer_sections(self) -> None:
        sections = 2.5
        with pytest.raises(AssertionError):
            list(chunker(self.samples, sections))



class TestConfidence:

    # source data contains 8 decimal places
    rel_central: float = 1e-7
    # The calculated confidence intervals in Minitab have only 4 decimal places
    rel_interval: float = 1e-3


    def test_confidence_to_alpha(self) -> None:
        level = 0.95
        alpha = confidence_to_alpha(level, two_sided=False, n_groups=1)
        assert alpha == approx(0.05)

        alpha = confidence_to_alpha(level, two_sided=True, n_groups=1)
        assert alpha == approx(0.025)

        alpha = confidence_to_alpha(level, two_sided=False, n_groups=5)
        assert alpha == approx(0.01)

        level = 0.9
        alpha = confidence_to_alpha(level, two_sided=True, n_groups=2)
        assert alpha == approx(0.025)
        
        with pytest.raises(AssertionError, match=r'\d+ not in \(0, 1\)'):
            confidence_to_alpha(5)

    def test_mean_ci(self) -> None:
        rows = ['mean', 'mean_95ci_low', 'mean_95ci_upp']
        
        size = 10
        for dist in df_dist10.columns:
            x_bar, ci_low, ci_upp = mean_ci(df_dist10[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = df_valid10[dist][rows]
            assert x_bar == approx(_x_bar, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)
        
        size = 25
        for dist in df_dist25.columns:
            x_bar, ci_low, ci_upp = mean_ci(df_dist25[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = df_valid25[dist][rows]
            assert x_bar == approx(_x_bar, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)

    def test_stdev_ci(self) -> None:
        rows = ['std', 'std_95ci_low', 'std_95ci_upp']
        
        size = 10
        for dist in df_dist10.columns:
            std, ci_low, ci_upp = stdev_ci(df_dist10[dist], level=.95)
            _std, _ci_low, _ci_upp = df_valid10[dist][rows]
            assert std == approx(_std, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)
        
        size = 25
        for dist in df_dist25.columns:
            std, ci_low, ci_upp = stdev_ci(df_dist25[dist], level=.95)
            _std, _ci_low, _ci_upp = df_valid25[dist][rows]
            assert std == approx(_std, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)

    def test_bonferroni_ci(self) -> None:
        data = pd.DataFrame({
            'target': np.random.normal(0, 1, 100),
            'feature': np.random.choice(['A', 'B', 'C'], 100)
        })
        # Test case 1: Check if the output DataFrame has the expected columns
        result = bonferroni_ci(data, 'target', 'feature')
        assert all(col in result.columns for col in ['midpoint', 'ci_low', 'ci_upp', 'feature'])

        # Test case 2: Check if the confidence intervals are within the correct range
        assert (result['ci_low'] <= result['midpoint']).all()
        assert (result['midpoint'] <= result['ci_upp']).all()

        # Test case 3: Check if the number of groups matches the expected value
        assert len(result) == len(data['feature'].unique())
    
    def test_bonferroni_ci_multiple_features(self) -> None:
        data = pd.DataFrame({
            'target': np.random.normal(0, 1, 100),
            'feature1': np.random.choice(['A', 'B', 'C'], 100),
            'feature2': np.random.choice(['X', 'Y', 'Z'], 100)
        })

        # Test case 1: Check if the output DataFrame has the expected columns
        result = bonferroni_ci(data, 'target', ['feature1', 'feature2'])
        assert all(col in result.columns for col in ['midpoint', 'ci_low', 'ci_upp', 'feature1', 'feature2'])

        # Test case 2: Check if the confidence intervals are within the correct range
        assert (result['ci_low'] <= result['midpoint']).all()
        assert (result['midpoint'] <= result['ci_upp']).all()

        # Test case 3: Check if the number of groups matches the expected value
        assert len(result) == len(data['feature1'].unique()) * len(data['feature2'].unique())


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