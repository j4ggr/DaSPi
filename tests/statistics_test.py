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

class TestConfidence:

    rel: float = 1e-3
    df_valid: pd.DataFrame = pd.read_csv(
        source/'dist_metrics_validation.csv', header=[0, 1], index_col=0)
    df_dist10: pd.DataFrame = pd.read_csv(source/f'dists_10-samples.csv')
    df_dist25: pd.DataFrame = pd.read_csv(source/f'dists_25-samples.csv')

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
        for dist in self.df_dist10.columns:
            x_bar, ci_low, ci_upp = mean_ci(self.df_dist10[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = self.df_valid[(dist, str(size))][rows]
            assert x_bar == approx(_x_bar, rel=self.rel)
            assert ci_low == approx(_ci_low, rel=self.rel)
            assert ci_upp == approx(_ci_upp, rel=self.rel)
        
        size = 25
        for dist in self.df_dist25.columns:
            x_bar, ci_low, ci_upp = mean_ci(self.df_dist25[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = self.df_valid[(dist, str(size))][rows]
            assert x_bar == approx(_x_bar, rel=self.rel)
            assert ci_low == approx(_ci_low, rel=self.rel)
            assert ci_upp == approx(_ci_upp, rel=self.rel)

    def test_stdev_ci(self):
        rows = ['std', 'std_95ci_low', 'std_95ci_upp']
        
        size = 10
        for dist in self.df_dist10.columns:
            x_bar, ci_low, ci_upp = stdev_ci(self.df_dist10[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = self.df_valid[(dist, str(size))][rows]
            assert x_bar == approx(_x_bar, rel=self.rel)
            assert ci_low == approx(_ci_low, rel=self.rel)
            assert ci_upp == approx(_ci_upp, rel=self.rel)
        
        size = 25
        for dist in self.df_dist25.columns:
            x_bar, ci_low, ci_upp = stdev_ci(self.df_dist25[dist], level=.95)
            _x_bar, _ci_low, _ci_upp = self.df_valid[(dist, str(size))][rows]
            assert x_bar == approx(_x_bar, rel=self.rel)
            assert ci_low == approx(_ci_low, rel=self.rel)
            assert ci_upp == approx(_ci_upp, rel=self.rel)
