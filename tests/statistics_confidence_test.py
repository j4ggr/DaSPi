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

from daspi.statistics.confidence import *


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

# source data contains 8 decimal places
rel_central: float = 1e-7
# The calculated confidence intervals in Minitab have only 4 decimal places
rel_interval: float = 1e-3


def test_confidence_to_alpha() -> None:
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

def test_mean_ci() -> None:
    rows = ['mean', 'mean_95ci_low', 'mean_95ci_upp']
    
    size = 10
    for dist in df_dist10.columns:
        x_bar, ci_low, ci_upp = mean_ci(df_dist10[dist], level=.95)
        _x_bar, _ci_low, _ci_upp = df_valid10[dist][rows]
        assert x_bar == approx(_x_bar, rel=rel_central)
        assert ci_low == approx(_ci_low, rel=rel_interval)
        assert ci_upp == approx(_ci_upp, rel=rel_interval)
    
    size = 25
    for dist in df_dist25.columns:
        x_bar, ci_low, ci_upp = mean_ci(df_dist25[dist], level=.95)
        _x_bar, _ci_low, _ci_upp = df_valid25[dist][rows]
        assert x_bar == approx(_x_bar, rel=rel_central)
        assert ci_low == approx(_ci_low, rel=rel_interval)
        assert ci_upp == approx(_ci_upp, rel=rel_interval)

def test_stdev_ci() -> None:
    rows = ['std', 'std_95ci_low', 'std_95ci_upp']
    
    size = 10
    for dist in df_dist10.columns:
        std, ci_low, ci_upp = stdev_ci(df_dist10[dist], level=.95)
        _std, _ci_low, _ci_upp = df_valid10[dist][rows]
        assert std == approx(_std, rel=rel_central)
        assert ci_low == approx(_ci_low, rel=rel_interval)
        assert ci_upp == approx(_ci_upp, rel=rel_interval)
    
    size = 25
    for dist in df_dist25.columns:
        std, ci_low, ci_upp = stdev_ci(df_dist25[dist], level=.95)
        _std, _ci_low, _ci_upp = df_valid25[dist][rows]
        assert std == approx(_std, rel=rel_central)
        assert ci_low == approx(_ci_low, rel=rel_interval)
        assert ci_upp == approx(_ci_upp, rel=rel_interval)

def test_bonferroni_ci() -> None:
    data = pd.DataFrame({
        'target': np.random.normal(0, 1, 100),
        'feature': np.random.choice(['A', 'B', 'C'], 100)
    })
    # Test case 1: Check if the output DataFrame has the expected columns
    result = bonferroni_ci(data, 'target', 'feature')
    assert all([c in result.columns for c in ['midpoint', 'lower', 'upper', 'feature']])

    # Test case 2: Check if the confidence intervals are within the correct range
    assert (result['lower'] <= result['midpoint']).all()
    assert (result['midpoint'] <= result['upper']).all()

    # Test case 3: Check if the number of groups matches the expected value
    assert len(result) == len(data['feature'].unique())

def test_bonferroni_ci_multiple_features() -> None:
    data = pd.DataFrame({
        'target': np.random.normal(0, 1, 100),
        'feature1': np.random.choice(['A', 'B', 'C'], 100),
        'feature2': np.random.choice(['X', 'Y', 'Z'], 100)
    })

    # Test case 1: Check if the output DataFrame has the expected columns
    result = bonferroni_ci(data, 'target', ['feature1', 'feature2'])
    assert all(col in result.columns for col in ['midpoint', 'lower', 'upper', 'feature1', 'feature2'])

    # Test case 2: Check if the confidence intervals are within the correct range
    assert (result['lower'] <= result['midpoint']).all()
    assert (result['midpoint'] <= result['upper']).all()

    # Test case 3: Check if the number of groups matches the expected value
    assert len(result) == len(data['feature1'].unique()) * len(data['feature2'].unique())

def test_confidence_to_alpha_edge_cases() -> None:
    # Test two-sided with n_groups > 1
    alpha = confidence_to_alpha(0.95, two_sided=True, n_groups=2)
    assert alpha == approx(0.0125)
    # Test one-sided with n_groups > 1
    alpha = confidence_to_alpha(0.95, two_sided=False, n_groups=2)
    assert alpha == approx(0.025)
    # Test invalid level
    with pytest.raises(AssertionError):
        confidence_to_alpha(-0.1)
    with pytest.raises(AssertionError):
        confidence_to_alpha(1.1)

def test_mean_ci_edge_cases() -> None:
    # Test with small sample
    arr = np.array([1.0, 2.0, 3.0])
    x_bar, ci_low, ci_upp = mean_ci(arr, level=0.99)
    assert x_bar == approx(np.mean(arr))
    assert ci_low < x_bar < ci_upp
    # Test with all identical values
    arr = np.ones(10)
    x_bar, ci_low, ci_upp = mean_ci(arr, level=0.95)
    assert x_bar == approx(1.0)
    assert pd.isna(ci_low)
    assert pd.isna(ci_upp)

def test_stdev_ci_edge_cases() -> None:
    # Test with small sample
    arr = np.array([1.0, 2.0, 3.0])
    std, ci_low, ci_upp = stdev_ci(arr, level=0.99)
    assert std == approx(np.std(arr, ddof=1))
    assert ci_low <= std <= ci_upp
    # Test with all identical values
    arr = np.ones(10)
    std, ci_low, ci_upp = stdev_ci(arr, level=0.95)
    assert std == approx(0.0)
    assert ci_low == approx(0.0)
    assert ci_upp == approx(0.0)

def test_bonferroni_ci_edge_cases() -> None:
    # Test with empty DataFrame
    df = pd.DataFrame({'target': [], 'feature': []})
    result = bonferroni_ci(df, 'target', 'feature')
    assert result.empty
    # Test with one group
    df = pd.DataFrame({'target': np.random.normal(0, 1, 10), 'feature': ['A']*10})
    result = bonferroni_ci(df, 'target', 'feature')
    assert len(result) == 1
    # Test with missing values
    df = pd.DataFrame({'target': [1, 2, np.nan, 4], 'feature': ['A', 'A', 'B', 'B']})
    result = bonferroni_ci(df, 'target', 'feature')
    assert all([c in result.columns for c in ['midpoint', 'lower', 'upper', 'feature']])

def test_proportion_ci_basic() -> None:
    # Typical case
    p, ci_low, ci_upp = proportion_ci(50, 100, level=0.95)
    assert 0 <= p <= 1
    assert ci_low <= p <= ci_upp
    # Edge case: zero events
    p, ci_low, ci_upp = proportion_ci(0, 100, level=0.95)
    assert p == 0
    assert ci_low <= p <= ci_upp
    # Edge case: all events
    p, ci_low, ci_upp = proportion_ci(100, 100, level=0.95)
    assert p == 1
    assert ci_low <= p <= ci_upp

def test_delta_mean_ci_basic() -> None:
    # Compare two samples
    a = np.random.normal(0, 1, 30)
    b = np.random.normal(1, 1, 30)
    delta, ci_low, ci_upp = delta_mean_ci(a, b, level=0.95)
    assert ci_low <= delta <= ci_upp

def test_delta_variance_ci_basic() -> None:
    a = np.random.normal(0, 1, 30)
    b = np.random.normal(0, 2, 30)
    delta, ci_low, ci_upp = delta_variance_ci(a, b, level=0.95)
    assert ci_low <= delta <= ci_upp

def test_delta_stdev_ci_basic() -> None:
    a = np.random.normal(0, 1, 30)
    b = np.random.normal(0, 2, 30)
    delta, ci_low, ci_upp = delta_stdev_ci(a, b, level=0.95)
    assert ci_low <= delta <= ci_upp

def test_delta_proportion_ci_basic() -> None:
    # Compare two proportions
    delta, ci_low, ci_upp = delta_proportions_ci(30, 100, 50, 100, level=0.95)
    assert ci_low <= delta <= ci_upp
    # Edge case: zero events in both
    delta, ci_low, ci_upp = delta_proportions_ci(0, 100, 0, 100, level=0.95)
    assert ci_low <= delta <= ci_upp

def test_delta_mean_ci_edge_cases() -> None:
    # Identical samples
    a = np.ones(10)
    b = np.ones(10)
    delta, ci_low, ci_upp = delta_mean_ci(a, b, level=0.95)
    assert delta == approx(0.0)
    assert ci_low == approx(0.0)
    assert ci_upp == approx(0.0)

def test_delta_stdev_ci_edge_cases() -> None:
    a = np.ones(10)
    b = np.ones(10)
    delta, ci_low, ci_upp = delta_stdev_ci(a, b, level=0.95)
    assert delta == approx(0.0)
    assert ci_low == approx(0.0)
    assert ci_upp == approx(0.0)

def test_delta_proportion_ci_edge_cases() -> None:
    # Identical proportions
    delta, ci_low, ci_upp = delta_proportions_ci(50, 100, 50, 100, level=0.95)
    assert delta == approx(0.0)
    assert ci_low <= delta <= ci_upp
