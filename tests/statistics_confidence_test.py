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