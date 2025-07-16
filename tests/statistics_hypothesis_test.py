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

from daspi.statistics.hypothesis import *

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
        samples = [1]
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
            list(chunker(self.samples, sections)) # type: ignore

# --- GROUPED TESTS FOR hypothesis.py COVERAGE ---
class TestHypothesisFunctions:
    def test_anderson_darling_test(self):
        from daspi.statistics.hypothesis import anderson_darling_test
        data = np.random.normal(0, 1, 100)
        p, stat = anderson_darling_test(data)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_all_normal(self):
        from daspi.statistics.hypothesis import all_normal
        a = np.random.normal(0, 1, 50)
        b = np.random.normal(0, 1, 50)
        assert all_normal(a, b)
        assert all_normal(a, b, p_threshold=0.01) in [True, False]
        with pytest.raises(AssertionError):
            all_normal(a, b, p_threshold=1.5)

    def test_kolmogorov_smirnov_test(self):
        from daspi.statistics.hypothesis import kolmogorov_smirnov_test
        data = np.random.normal(0, 1, 100)
        p, D, params = kolmogorov_smirnov_test(data, 'norm')
        assert isinstance(p, float)
        assert isinstance(D, float)
        assert isinstance(params, tuple)
        assert 0 <= p <= 1

    def test_f_test(self) -> None:
        from daspi.statistics.hypothesis import f_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, F = f_test(a, b)
        assert isinstance(p, float)
        assert isinstance(F, float)
        assert 0 <= p <= 1

    def test_t_test(self) -> None:
        from daspi.statistics.hypothesis import t_test
        a = np.random.normal(0, 1, 30)
        p, t, df = t_test(a)
        assert isinstance(p, float)
        assert isinstance(t, float)
        assert isinstance(df, int)
        assert 0 <= p <= 1

    def test_levene_test(self) -> None:
        from daspi.statistics.hypothesis import levene_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, L = levene_test(a, b)
        assert isinstance(p, float)
        assert isinstance(L, float)
        assert 0 <= p <= 1

    def test_variance_stability_test(self) -> None:
        from daspi.statistics.hypothesis import variance_stability_test
        a = np.random.normal(0, 1, 30)
        p, L = variance_stability_test(a, n_sections=3)
        assert isinstance(p, float)
        assert isinstance(L, float)
        assert 0 <= p <= 1

    def test_mean_stability_test(self) -> None:
        from daspi.statistics.hypothesis import mean_stability_test
        a = np.random.normal(0, 1, 30)
        p, stat = mean_stability_test(a, n_sections=3)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_position_test(self) -> None:
        from daspi.statistics.hypothesis import position_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, stat, test = position_test(a, b)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert isinstance(test, str)
        assert 0 <= p <= 1

    def test_variance_test(self) -> None:
        from daspi.statistics.hypothesis import variance_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, stat, test = variance_test(a, b)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert isinstance(test, str)
        assert 0 <= p <= 1

    def test_proportions_test(self) -> None:
        from daspi.statistics.hypothesis import proportions_test
        p, stat, test = proportions_test(5, 10, 7, 12)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert isinstance(test, str)
        assert 0 <= p <= 1

    def test_kurtosis_test(self) -> None:
        from daspi.statistics.hypothesis import kurtosis_test
        a = np.random.normal(0, 1, 30)
        p, stat = kurtosis_test(a)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_skew_test(self) -> None:
        from daspi.statistics.hypothesis import skew_test
        a = np.random.normal(0, 1, 30)
        p, stat = skew_test(a)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_ensure_generic(self) -> None:
        from daspi.statistics.hypothesis import ensure_generic
        dist = ensure_generic('norm')
        assert hasattr(dist, 'cdf')
        import scipy.stats as stats
        dist2 = ensure_generic(stats.norm)
        assert hasattr(dist2, 'cdf')
        with pytest.raises(AttributeError):
            ensure_generic('not_a_dist')