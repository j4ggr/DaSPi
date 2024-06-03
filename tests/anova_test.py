import sys
import patsy
import patsy.design_info
import patsy.highlevel
import pytest

import numpy as np
import pandas as pd

from pathlib import Path
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from pandas.core.frame import DataFrame
from statsmodels.regression.linear_model import RegressionResultsWrapper

sys.path.append(str(Path(__file__).parent.resolve()))

import daspi

from daspi.anova import uniques
from daspi.anova import hierarchical
from daspi.anova import get_term_name
from daspi.anova import is_main_feature

from daspi.anova import LinearModel

valid_data_dir = Path(__file__).parent/'data'


class TestHierarchical:

    def test_basics(self) -> None:
        expected_output = ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'A:B:C']
        assert hierarchical(['A:B:C']) == expected_output

    def test_empty(self) -> None:
        assert hierarchical([]) == []

    def test_single_feature(self) -> None:
        assert hierarchical(['A']) == ['A']

    def test_duplicate_features(self) -> None:
        assert hierarchical(['A', 'A']) == ['A']

    def test_multiple_duplicate_features(self) -> None:
        assert hierarchical(['A', 'A', 'B', 'B']) == ['A', 'B']

    def test_interactions(self) -> None:
        assert hierarchical(['A:B', 'B:C']) == ['A', 'B', 'C', 'A:B', 'B:C']


class TestGetTermName:

    def test_no_interaction(self) -> None:
        encoded_name = 'x1[T.b]'
        term_name = get_term_name(encoded_name)
        assert term_name == 'x1'

    def test_with_interaction(self) -> None:
        encoded_name = 'x1[T.b]:x2[T.2]'
        term_name = get_term_name(encoded_name)
        assert term_name == 'x1:x2'

    def test_multiple_interactions(self) -> None:
        encoded_name = 'x1[T.b]:x2[T.2]:x3[T.True]'
        term_name = get_term_name(encoded_name)
        assert term_name == 'x1:x2:x3'

    def test_no_encoding(self) -> None:
        term_name = get_term_name('Category')
        assert term_name == 'Category'

    def test_empty_string(self) -> None:
        term_name = get_term_name('')
        assert term_name == ''

    def test_invalid_encoding(self) -> None:
        term_name = get_term_name('InvalidEncoding')
        assert term_name == 'InvalidEncoding'


class TestIsMainFeature:

    def test_basics(self) -> None:
        assert is_main_feature('A') == True
        assert is_main_feature('B') == True
        assert is_main_feature('Intercept') == False
        assert is_main_feature('A:B') == False
        assert is_main_feature('A:B:C') == False

    def test_empty(self) -> None:
        assert is_main_feature('') == True

    def test_whitespace(self) -> None:
        assert is_main_feature(' ') == True
        assert is_main_feature('  ') == True

    def test_separator(self) -> None:
        assert is_main_feature(':') == False
        assert is_main_feature('A:') == False
        assert is_main_feature('A:B') == False
        assert is_main_feature('A:B:C') == False


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


@pytest.fixture
def lm() -> LinearModel:
    """LinearModel no covariates, simple"""
    source = pd.DataFrame({
        'A': [1, 0, 1, 1, 0, 1],
        'B': [-1, 1, 0, -1, 1, 0],
        'C': [3.1, 6.1, 9.1, 12.0, 15.0, 18.1],
        'bad': [0, 0, 0, 0, 0, 0],
        'Target': [11, 19, 30, 42, 49, 50]})
    target = 'Target'
    features = ['A', 'B', 'C', 'bad']
    covariates = []
    alpha = 0.05
    return LinearModel(source, target, features, covariates, alpha, False)

@pytest.fixture
def lm2() -> LinearModel:
    """LinearModel no covariates, complete"""
    source = pd.DataFrame({
        'A': [1, 0, 1, 1, 0, 1],
        'B': [-1, 1, 0, -1, 1, 0],
        'C': [3.1, 6.1, 9.1, 12.0, 15.0, 18.1],
        'bad': [0, 0, 0, 0, 0, 0],
        'Target': [11, 19, 30, 42, 49, 50]})
    target = 'Target'
    features = ['A', 'B', 'C', 'bad']
    covariates = []
    alpha = 0.05
    return LinearModel(source, target, features, covariates, alpha, True)

@pytest.fixture
def lm3() -> LinearModel:
    """LinearModel with covariates, simple"""
    source = pd.DataFrame({
        'A': ['a', 'b', 'c', 'a', 'b', 'c'],
        'B': [1, 2, 3, 1, 2, 3],
        'C': [True, False, True, False, True, False],
        'D': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Target': [10, 20, 30, 40, 50, 60]})
    target = 'Target'
    features = ['A', 'B', 'C']
    covariates = ['D']
    alpha = 0.05
    return LinearModel(source, target, features, covariates, alpha, False)

@pytest.fixture
def lm4() -> LinearModel:
    """LinearModel with covariates, complete"""
    source = pd.DataFrame({
        'A': ['a', 'b', 'c', 'a', 'b', 'c'],
        'B': [1, 2, 3, 1, 2, 3],
        'C': [True, False, True, False, True, False],
        'D': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Target': [10, 20, 30, 40, 50, 60]})
    target = 'Target'
    features = ['A', 'B', 'C']
    covariates = ['D']
    alpha = 0.05
    return LinearModel(source, target, features, covariates, alpha, True)

@pytest.fixture
def anova3_c_valid() -> DataFrame:
    df = pd.read_csv(
        valid_data_dir/'anova3_result.csv', skiprows=9, sep=';', index_col=0)
    return df

@pytest.fixture
def anova3_s_valid() -> DataFrame:
    df = pd.read_csv(
        valid_data_dir/'anova3_result.csv', skiprows=1, skipfooter=12, sep=';',
        index_col=0)
    return df


class TestLinearModel:

    def test_init(self, lm: LinearModel) -> None:
        assert_frame_equal(lm.data, pd.DataFrame({
            'x0': [1, 0, 1, 1, 0, 1],
            'x1': [-1, 1, 0, -1, 1, 0],
            'x2': [3.1, 6.1, 9.1, 12.0, 15.0, 18.1],
            'x3': [0, 0, 0, 0, 0, 0],
            'y': [11, 19, 30, 42, 49, 50]}))
        assert lm.target == 'Target'
        assert lm.features == ['A', 'B', 'C', 'bad']
        assert lm.covariates == []
        assert lm.alpha == 0.05
        assert lm.output_map == {'Target': 'y'}
        assert lm.input_map == {'A': 'x0', 'B': 'x1', 'C': 'x2', 'bad': 'x3'}
        assert lm.gof_metrics == {}
        assert_frame_equal(lm.dmatrix, pd.DataFrame())
        assert lm.exclude == set()
        assert lm._model is None
        assert lm.gof_metrics == {}
    
    def test_formula(self, lm: LinearModel, lm2: LinearModel) -> None:
        assert '*' not in lm.formula
        assert '+' in lm.formula
        assert ':' not in lm.formula
        lm.fit()
        assert '*' not in lm.formula
        assert '+' in lm.formula
        assert ':' not in lm.formula

        assert '*' in lm2.formula
        assert '+' not in lm2.formula
        assert ':' not in lm2.formula
        lm2.fit()
        assert '*' not in lm2.formula
        assert '+' in lm2.formula
        assert ':' in lm2.formula
        lm2.exclude.add('Intercept')
        assert '-1' in lm2.formula

    def test_model_property(self, lm: LinearModel) -> None:
        with pytest.raises(AssertionError):
            lm.model
        lm.fit()
        assert isinstance(lm.model, RegressionResultsWrapper)

    def test_least_term(self, lm: LinearModel, lm4: LinearModel) -> None:
        lm4.fit()
        assert all(lm4.p_values.isna())
        assert lm4.least_term() == 'B:C'

        lm.fit()
        assert any(lm.p_values.isna())
        assert lm.p_values.max() > 0.05
        assert lm.least_term() == 'bad'

    def test_main_features_property(self, lm2: LinearModel) -> None:
        lm2.fit()
        assert lm2.main_features == ['x0', 'x1', 'x2', 'x3'] # order changes because x0 and x2 are categoricals. patsy set them in front of the non categoricals
        lm2.recursive_feature_elimination()
        assert lm2.main_features == ['x0', 'x2']

    def test_alpha_property(self, lm: LinearModel) -> None:
        assert lm.alpha == 0.05
        lm.alpha = 0.1
        assert lm.alpha == 0.1
        with pytest.raises(AssertionError):
            lm.alpha = -0.1
        with pytest.raises(AssertionError):
            lm.alpha = 1.1
    
    def test_anova(self, anova3_s_valid: DataFrame, anova3_c_valid: DataFrame) -> None:
        df = daspi.load_dataset('anova3')
        lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'])
        
        anova = lm.fit().anova()
        valid = anova3_s_valid
        assert_series_equal(
            anova['df'], valid['df'], check_dtype=False)
        assert_series_equal(
            anova['sum_sq'], valid['sum_sq'], check_exact=False, atol=1e-2)

        lm = LinearModel(
            df, 'Cholesterol', ['Sex', 'Risk', 'Drug'], complete=True)
        anova = lm.fit().anova()
        valid = anova3_c_valid
        assert_series_equal(
            anova['df'], valid['df'], check_dtype=False)
        assert_series_equal(
            anova['sum_sq'], valid['sum_sq'], check_exact=False, atol=1e-2)

