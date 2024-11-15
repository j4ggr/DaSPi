import sys
import pytest

import pandas as pd

from pathlib import Path

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi import BivariateUnivariateCharts


class TestBivariateUnivariateChart:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        cat = ['A', 'A', 'B', 'B', 'A', 'B']
        return pd.DataFrame({
            'x': [11.5, 8.5, 9, 11, 8, 10.5]*3,
            'z': [1.1, 0.9, 1.0, 1.2, 0.8, 1.0]*3,
            'category': cat*2 + cat[::-1],
        })
    
    def test_init(self, sample_data: pd.DataFrame) -> None:
        chart = BivariateUnivariateCharts(
            sample_data,
            feature='x',
            target='y',
            hue='category')
        assert chart.features == ('', '', 'x', '')
        assert chart.targets == ('x', '', 'y', 'y')
        assert chart.hues == ('category', 'category', 'category', 'category')
        assert chart.target_on_ys == (False, True, True, True)
        assert chart.n_axes == 4
        assert chart.dodges == tuple([False]*4)
        assert chart.categorical_features == tuple([False]*4)

        chart = BivariateUnivariateCharts(
            sample_data,
            feature='x',
            target='y',
            hue='category',
            dodge_univariates=True)
        assert chart.n_axes == 4
        assert chart.features == ('', '', 'x', '')
        assert chart.targets == ('x', '', 'y', 'y')
        assert chart.hues == ('category', 'category', 'category', 'category')
        assert chart.target_on_ys == (False, True, True, True)
        assert chart.dodges == (True, False, False, True)
        assert chart.categorical_features == chart.dodges

