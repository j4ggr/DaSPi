import sys
import patsy.design_info
import patsy.highlevel
import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pytest import approx
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi import ANOVA
from daspi.anova.tables import *

valid_data_dir = Path(__file__).parent/'data'


class TestUniques:

    def test_basics(self) -> None:
        sequence = [1, 2, 3, 2, 1, 4, 5, 4]
        unique_elements = uniques(sequence)
        assert unique_elements == [1, 2, 3, 4, 5]

    def test_empty_sequence(self) -> None:
        sequence = []
        unique_elements = uniques(sequence)
        assert unique_elements == []

    def test_single_element(self) -> None:
        sequence = [1]
        unique_elements = uniques(sequence)
        assert unique_elements == [1]

    def test_all_duplicates(self) -> None:
        sequence = [1, 1, 1, 1, 1]
        unique_elements = uniques(sequence)
        assert unique_elements == [1]

    def test_mixed_types(self) -> None:
        sequence = [1, 'a', 'b', 2, 'a', 3, 2, 1]
        unique_elements = uniques(sequence)
        assert unique_elements == [1, 'a', 'b', 2, 3]
    
    def test_dict_values(self) -> None:
        sequence = [1, 'a', 'b', 2, 'a', 3, 2, 1]
        _dict = {i: v for i, v in enumerate(sequence)}
        unique_elements = uniques(_dict.values())
        assert unique_elements == [1, 'a', 'b', 2, 3]


class TestVarianceInflationFactor:

    X = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], columns=['x1', 'x2'])
    y = pd.Series([1, 2, 3, 4])
    n_cols = len(ANOVA.VIF_COLNAMES)

    def test_two_predictors(self) -> None:
        model = sm.OLS(self.y, self.X).fit()
        vif_table = variance_inflation_factor(model)
        assert vif_table.shape == (2, self.n_cols)
        assert vif_table.loc['x1', 'Method'] == 'R_squared'
        assert vif_table.loc['x2', 'Method'] == 'R_squared'
        assert vif_table.loc['x1', 'VIF'] == 1.0
        assert vif_table.loc['x2', 'VIF'] == 1.0

    def test_collinear_predictors(self) -> None:
        X = self.X.copy()
        X['x3'] = X['x2']
        model = sm.OLS(self.y, X).fit()
        vif_table = variance_inflation_factor(model)
        assert vif_table.shape == (3, self.n_cols)
        assert not vif_table.loc['x1', 'Collinear']
        assert vif_table.loc['x2', 'Collinear']
        assert vif_table.loc['x3', 'Collinear']
        assert vif_table.loc['x1', 'VIF'] == 1.0
        assert vif_table.loc['x2', 'VIF'] == float('inf')
        assert vif_table.loc['x3', 'VIF'] == float('inf')

    def test_categorical_predictors(self) -> None:
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5, 6],
            'x1': ['a', 'b', 'c', 'a', 'b', 'c'],
            'x2': ['I', 'II', 'I', 'II', 'I', 'II']})
        model = smf.ols('y ~ x1*x2', data).fit()
        vif_table = variance_inflation_factor(model)
        assert vif_table.shape == (4, self.n_cols)
        assert vif_table.loc[ANOVA.INTERCEPT, 'Method'] == 'R_squared'
        assert vif_table.loc['x1', 'Method'] == 'generalized'
        assert vif_table.loc['x2', 'Method'] == 'R_squared'
        assert vif_table.loc['x1:x2', 'Method'] == 'single_order-2_term'
        assert vif_table.loc[ANOVA.INTERCEPT, 'VIF'] == approx(4.0)
        assert vif_table.loc['x1', 'VIF'] == approx(1.0)
        assert vif_table.loc['x2', 'VIF'] == approx(1.0)
        assert vif_table.loc['x1:x2', 'VIF'] == approx(1.0)
        assert not vif_table.loc[ANOVA.INTERCEPT, 'Collinear']
        assert not vif_table.loc['x1', 'Collinear']
        assert not vif_table.loc['x2', 'Collinear']
        assert not vif_table.loc['x1:x2', 'Collinear']

    def test_single_predictor(self) -> None:
        # Test case with single predictor variable
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 2, 3, 4])
        with pytest.raises(AssertionError):
            variance_inflation_factor(sm.OLS(y, X).fit())
