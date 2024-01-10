import sys
import pytest

import numpy as np
import pandas as pd

from pytest import approx
from pathlib import Path
from numpy.typing import ArrayLike 

sys.path.append(Path(__file__).parent.resolve())

from daspi.statistics.confidence import *
from daspi.statistics.estimation import *
from daspi.statistics.hypothesis import *

source = Path(__file__).parent/'data'
df_dist10: pd.DataFrame = pd.read_csv(source/f'dists_10-samples.csv')
df_dist25: pd.DataFrame = pd.read_csv(source/f'dists_25-samples.csv')
df_valid: pd.DataFrame = pd.read_csv(
    source/'dist_metrics_validation.csv', header=[0, 1], index_col=0)

class TestConfidence:

    rel_central: float = 1e-7
    # The calculated confidence intervals in Minitab have only 4 decimal places
    rel_interval: float = 1e-3 

    def test_confidence_to_alpha(self):
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

    def test_mean_ci(self):
        rows = ['mean', 'mean_95ci_low', 'mean_95ci_upp']
        
        size = 10
        for dist in df_dist10.columns:
            x_bar, ci_low, ci_upp = mean_ci(df_dist10[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = df_valid[(dist, str(size))][rows]
            assert x_bar == approx(_x_bar, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)
        
        size = 25
        for dist in df_dist25.columns:
            x_bar, ci_low, ci_upp = mean_ci(df_dist25[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = df_valid[(dist, str(size))][rows]
            assert x_bar == approx(_x_bar, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)

    def test_stdev_ci(self):
        rows = ['std', 'std_95ci_low', 'std_95ci_upp']
        
        size = 10
        for dist in df_dist10.columns:
            std, ci_low, ci_upp = stdev_ci(df_dist10[dist], level=.95)
            _std, _ci_low, _ci_upp = df_valid[(dist, str(size))][rows]
            assert std == approx(_std, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)
        
        size = 25
        for dist in df_dist25.columns:
            std, ci_low, ci_upp = stdev_ci(df_dist25[dist], level=.95)
            _std, _ci_low, _ci_upp = df_valid[(dist, str(size))][rows]
            assert std == approx(_std, rel=self.rel_central)
            assert ci_low == approx(_ci_low, rel=self.rel_interval)
            assert ci_upp == approx(_ci_upp, rel=self.rel_interval)


class TestEstimation:

    estimate: Estimator = Estimator(df_dist10['rayleigh'])

    def test_data_filtered(self):
        N = 10
        N_nan = 2
        data = np.concatenate((np.random.randn(N-2), N_nan*[np.nan], [.1, -.1]))
        estimate = Estimator(data)
        assert len(estimate.data) == N + N_nan
        assert estimate._filtered.empty
        assert len(estimate.filtered) == N
        assert not estimate._filtered.empty
        assert not any(np.isnan(estimate.filtered))

    def test_mean(self):
        assert self.estimate._mean is None
        assert self.estimate.mean == approx(np.mean(self.estimate.filtered))
        assert self.estimate._mean is not None

    def test_median(self):
        assert self.estimate._median is None
        assert self.estimate.median == approx(np.median(self.estimate.filtered))
        assert self.estimate._median is not None

    def test_std(self):
        assert self.estimate._std is None
         # do not remove ddof=1, numpy uses ddof=0 as default!
        assert self.estimate.std == approx(np.std(self.estimate.filtered, ddof=1))
        assert self.estimate._std is not None

    def test_excess(self):
        
        size = 10
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist10[dist])
            excess = estimate.excess
            assert excess == df_valid[(dist, str(size))]['excess']
        
        size = 25
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist25[dist])
            excess = estimate.excess
            assert excess == df_valid[(dist, str(size))]['excess']

    def test_skew(self):
        
        size = 10
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist10[dist])
            skew = estimate.skew
            assert skew == df_valid[(dist, str(size))]['skew']
        
        size = 25
        for dist in df_dist10.columns:
            estimate = Estimator(df_dist25[dist])
            skew = estimate.skew
            assert skew == df_valid[(dist, str(size))]['skew']

    def test_follows_norm_curve(self):
        estimate = Estimator(df_dist25['norm'])
        assert estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['foldnorm'])
        assert not estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['chi2'])
        assert not estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['gamma'])
        assert not estimate.follows_norm_curve()
        
        estimate = Estimator(df_dist25['wald'])
        assert not estimate.follows_norm_curve()

        estimate = Estimator(df_dist25['expon'])
        assert not estimate.follows_norm_curve()
