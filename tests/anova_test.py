import sys
import patsy
import patsy.design_info
import patsy.highlevel
import pytest

import numpy as np
import pandas as pd

from patsy.highlevel import dmatrices
from typing import List
from pytest import approx
from pathlib import Path
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from statsmodels.regression.linear_model import RegressionResultsWrapper

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi.anova import decode
from daspi.anova import optimize
from daspi.anova import hierarchical
from daspi.anova import get_term_name
from daspi.anova import is_main_feature
from daspi.anova import decode_cat_main
from daspi.anova import encoded_dmatrices
from daspi.anova import prepare_encoding_data
from daspi.anova import is_encoded_categorical
from daspi.anova import clean_categorical_names
from daspi.anova import remove_special_characters

from daspi.anova import LinearModel


class TestHierarchical:

    def test_hierarchical(self) -> None:
        expected_output = ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'A:B:C']
        assert hierarchical(['A:B:C']) == expected_output

    def test_hierarchical_empty(self) -> None:
        assert hierarchical([]) == []

    def test_hierarchical_single_feature(self) -> None:
        assert hierarchical(['A']) == ['A']

    def test_hierarchical_duplicate_features(self) -> None:
        assert hierarchical(['A', 'A']) == ['A']

    def test_hierarchical_multiple_duplicate_features(self) -> None:
        assert hierarchical(['A', 'A', 'B', 'B']) == ['A', 'B']

    def test_hierarchical_interactions(self) -> None:
        assert hierarchical(['A:B', 'B:C']) == ['A', 'B', 'C', 'A:B', 'B:C']


class TestGetTermName:

    def test_get_term_name(self) -> None:
        assert get_term_name('A') == 'A'
        assert get_term_name('A[T.b]') == 'A'


class TestIsMainFeature:

    def test_is_main_feature(self) -> None:
        assert is_main_feature('A') == True
        assert is_main_feature('B') == True
        assert is_main_feature('Intercept') == False
        assert is_main_feature('A:B') == False
        assert is_main_feature('A:B:C') == False

    def test_is_main_feature_empty(self) -> None:
        assert is_main_feature('') == True

    def test_is_main_feature_whitespace(self) -> None:
        assert is_main_feature(' ') == True
        assert is_main_feature('  ') == True

    def test_is_main_feature_separator(self) -> None:
        assert is_main_feature(':') == False
        assert is_main_feature('A:') == False
        assert is_main_feature('A:B') == False
        assert is_main_feature('A:B:C') == False


class TestDecodeCatMain:

    X = None

    @property
    def di(self) -> patsy.design_info.DesignInfo:
        if self.X is None:
            self.data = pd.DataFrame({
                'A': ['a', 'b', 'c', 'a', 'b', 'c'],
                'B': [1, 2, 3, 1, 2, 3],
                'C': [True, False, True, False, True, False],
                'D': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'Target': [10, 20, 30, 40, 50, 60]})
            formula = 'Target ~ A * B * C * D'
            y, self.X = patsy.highlevel.dmatrices(
                formula, self.data, return_type='dataframe')
        return self.X.design_info

    def test_decode_cat_main(self) -> None:
        assert decode_cat_main('A[T.b]', 0, self.di) == 'a | c'
        assert decode_cat_main('A[T.b]', 1, self.di) == 'b'
        assert decode_cat_main('A[T.c]', 0, self.di) == 'a | b'
        assert decode_cat_main('A[T.c]', 1, self.di) == 'c'
        assert decode_cat_main('C[T.True]', 0, self.di) == 'False'
        assert decode_cat_main('C[T.True]', 1, self.di) == 'True'


class TestEncodedDmatrices:

    @property
    def data(self) -> DataFrame:
        return pd.DataFrame({
            'A': ['a', 'b', 'c', 'a', 'b', 'c'],
            'B': [1, 2, 3, 1, 2, 3],
            'C': [True, False, True, False, True, False],
            'D': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Target': [10, 20, 30, 40, 50, 60]
        })

    def test_encoded_dmatrices(self) -> None:
        formula = 'Target ~ A + B + C + D + A:B + A:C'
        expected_y = pd.DataFrame({'Target': [10, 20, 30, 40, 50, 60]})
        expected_X_code = pd.DataFrame({
            'Intercept': [1, 1, 1, 1, 1, 1],
            'A_b': [0, 1, 0, 0, 1, 0],
            'A_c': [0, 0, 1, 0, 0, 1],
            'B': [-1, 0, 1, -1, 0, 1],
            'C_True': [1, 0, 1, 0, 1, 0],
            'D': [-1, -0.6, -0.2, 0.2, 0.6, 1],
            'A_b:B': [0, 0, 0, 0, 0, 0],
            'A_c:B': [0, 0, 1, 0, 0, 1],
            'A_b:C_True': [0, 0, 0, 0, 1, 0],
            'A_c:C_True': [0, 0, 1, 0, 0, 0]})
        
        expected_mapper = {
            'A_b': {'a | c': 0, 'b': 1},
            'A_c': {'a | b': 0, 'c': 1},
            'C_True': {'False': 0, 'True': 1},
            'B': {1.0: -1.0, 2.0: 0.0, 3.0: 1.0},
            'D': {1.0: -1.0, 2.0: -0.6, 3.0: -0.2, 4.0: 0.2, 5.0: 0.6, 6.0: 1.0}}

        y, X_code, mapper = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code[X_code.columns], check_dtype=False)
        for key, value in expected_mapper.items():
            assert approx(mapper[key]) == value

    def test_encoded_dmatrices_empty(self) -> None:
        formula = 'Target ~ 1'
        expected_y = pd.DataFrame({'Target': [10, 20, 30, 40, 50, 60]})
        expected_X_code = pd.DataFrame({'Intercept': [1, 1, 1, 1, 1, 1]})
        expected_mapper = {}
        y, X_code, mapper = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code, check_dtype=False)
        assert mapper == expected_mapper

    def test_encoded_dmatrices_single_feature(self) -> None:
        formula = 'Target ~ A'
        expected_y = pd.DataFrame({'Target': [10, 20, 30, 40, 50, 60]})
        expected_X_code = pd.DataFrame({
            'Intercept': [1, 1, 1, 1, 1, 1],
            'A_b': [0, 1, 0, 0, 1, 0],
            'A_c': [0, 0, 1, 0, 0, 1]})
        
        expected_mapper = {
            'A_b': {'a | c': 0, 'b': 1},
            'A_c': {'a | b': 0, 'c': 1}}
        
        y, X_code, mapper = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code, check_dtype=False)
        assert mapper == expected_mapper

    def test_encoded_dmatrices_no_categorical(self) -> None:
        formula = 'Target ~ B + D'
        expected_y = pd.DataFrame({'Target': [10, 20, 30, 40, 50, 60]})
        expected_X_code = pd.DataFrame({
            'Intercept': [1, 1, 1, 1, 1, 1],
            'B': [-1, 0, 1, -1, 0, 1],
            'D': [-1, -0.6, -0.2, 0.2, 0.6, 1]})

        expected_mapper = {
            'B': {1.0: -1.0, 2.0: 0.0, 3.0: 1.0},
            'D': {1.0: -1.0, 2.0: -0.6, 3.0: -0.2, 4.0: 0.2, 5.0: 0.6, 6.0: 1.0}}

        y, X_code, mapper = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code, check_dtype=False)
        for key, value in expected_mapper.items():
            assert approx(mapper[key]) == value


class TestCleanCategoricalNames:

    def test_clean_categorical_names(self) -> None:
        assert clean_categorical_names('A') == 'A'
        assert clean_categorical_names('A[T.b]') == 'A_b'
        assert clean_categorical_names('A[T.b]:C[T.d]') == 'A_b:C_d'
        assert clean_categorical_names('A[T.b]:C[T.d]:D') == 'A_b:C_d:D'
        assert clean_categorical_names('A[T.b]:C[T.d]:D[T.e]') == 'A_b:C_d:D_e'
        assert clean_categorical_names('A[T.b]:C') == 'A_b:C'
        assert clean_categorical_names('A[T.b]:C[T.d]:D[T.e]:E') == 'A_b:C_d:D_e:E'

    def test_clean_categorical_names_no_encoding(self) -> None:
        assert clean_categorical_names('A') == 'A'
        assert clean_categorical_names('A:B') == 'A:B'
        assert clean_categorical_names('A:B:C') == 'A:B:C'

    def test_clean_categorical_names_empty(self) -> None:
        assert clean_categorical_names('') == ''

    def test_clean_categorical_names_whitespace(self) -> None:
        assert clean_categorical_names(' ') == ' '
        assert clean_categorical_names('  ') == '  '

    def test_clean_categorical_names_no_match(self) -> None:
        assert clean_categorical_names('A[T]') == 'A[T]'
        assert clean_categorical_names('A[T.b') == 'A[T.b'
        assert clean_categorical_names('A[T.b]:C[T') == 'A_b:C[T'



@pytest.fixture
def linear_model() -> LinearModel:
    source = pd.DataFrame({
        'A': [1, 0, 0, 1, 0],
        'B': [2, 4, 6, 8, 10],
        'C': [3.1, 6.1, 9.1, 12.0, 15.0],
        'Null': [0, 0, 0, 0, 0],
        'Target': [11, 19, 30, 42, 49]
    })
    target = 'Target'
    features = ['A', 'B', 'C']
    covariates = []
    alpha = 0.05
    return LinearModel(source, target, features, covariates, alpha)


class TestLinearModel:

    def test_linear_model_init(self, linear_model: LinearModel) -> None:
        assert_frame_equal(linear_model.source, pd.DataFrame({
            'A': [1, 0, 0, 1, 0],
            'B': [2, 4, 6, 8, 10],
            'C': [3.1, 6.1, 9.1, 12.0, 15.0],
            'Null': [0, 0, 0, 0, 0],
            'Target': [11, 19, 30, 42, 49]
        }))
        assert linear_model.target == 'Target'
        assert linear_model.features == ['A', 'B', 'C']
        assert linear_model.covariates == []
        assert linear_model.alpha == 0.05
        assert linear_model.output_map == {'Target': 'y'}
        assert linear_model.input_map == {'A': 'x0', 'B': 'x1', 'C': 'x2'}
        assert linear_model.gof_metrics == {}
        assert_frame_equal(linear_model.dmatrix, pd.DataFrame())
        assert linear_model.exclude == set()
        assert linear_model._model is None
        assert linear_model.gof_metrics == {}

    def test_linear_model_model_property(self, linear_model: LinearModel) -> None:
        with pytest.raises(AssertionError):
            linear_model.model
        linear_model.fit()
        assert isinstance(linear_model.model, RegressionResultsWrapper)

    def test_linear_model_p_values_property(self, linear_model: LinearModel) -> None:
        linear_model.fit()
        assert_series_equal(linear_model.p_values, pd.Series([0.1, 0.2, 0.3]))

    def test_linear_model_p_least_property(self, linear_model: LinearModel) -> None:
        linear_model.fit()
        assert linear_model.p_least == 0.3

def test_linear_model_main_features_property(linear_model: LinearModel) -> None:
    linear_model.exclude = {'A'}
    assert linear_model.main_features == ['B', 'C']

def test_linear_model_alpha_property(linear_model: LinearModel) -> None:
    assert linear_model.alpha == 0.05
    linear_model.alpha = 0.1
    assert linear_model.alpha == 0.1
    with pytest.raises(AssertionError):
        linear_model.alpha = -0.1
    with pytest.raises(AssertionError):
        linear_model.alpha = 1.1

def test_linear_model_endogenous_property(linear_model: LinearModel) -> None:
    assert linear_model.endogenous == 'y'

def test_linear_model_exogenous_property(linear_model):
    linear_model.dmatrix = pd.DataFrame({
        'y': [10, 20, 30, 40, 50],
        'x0': [1, 2, 3, 4, 5],
        'x1': [2, 4, 6, 8, 10],
        'x2': [3, 6, 9, 12, 15]
    })
    linear_model.exclude = {'x0'}
    assert linear_model.exogenous == ['x1', 'x2']

def test_linear_model_formula_property(linear_model):
    linear_model.dmatrix = pd.DataFrame({
        'y': [10, 20, 30, 40, 50],
        'x0': [1, 2, 3, 4, 5],
        'x1': [2, 4, 6, 8, 10],
        'x2': [3, 6, 9, 12, 15]
    })
    linear_model.exclude = {'x0'}
    assert linear_model.formula == 'y~x1+x2'

def test_linear_model_construct_design_matrix(linear_model):
    source = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],
        'C': [3, 6, 9, 12, 15],
        'Target': [10, 20, 30, 40, 50]
    })
    target = 'Target'
    features = ['A', 'B', 'C']
    covariates = []
    alpha = 0.05
    linear_model = LinearModel(source, target, features, covariates, alpha)
    linear_model.construct_design_matrix()
    assert_frame_equal(linear_model.dmatrix, pd.DataFrame({
        'y': [10, 20, 30, 40, 50],
        'x0': [1, 2, 3, 4, 5],
        'x1': [2, 4, 6, 8, 10],
        'x2': [3, 6, 9, 12, 15]
    }))

def test_linear_model_construct_design_matrix_complete(linear_model):
    source = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],
        'C': [3, 6, 9, 12, 15],
        'Target': [10, 20, 30, 40, 50]
    })
    target = 'Target'
    features = ['A', 'B', 'C']
    covariates = []
    alpha = 0.05
    linear_model = LinearModel(source, target, features, covariates, alpha)
    linear_model.construct_design_matrix(encode=True, complete=True)
    assert_frame_equal(linear_model.dmatrix, pd.DataFrame({
        'y': [10, 20, 30, 40, 50],
        'x0': [1, 2, 3, 4, 5],
        'x1': [2, 4, 6, 8, 10],
        'x2': [3, 6, 9, 12, 15],
        'x0:x1': [2, 8, 18, 32, 50],
        'x0:x2': [3, 12, 27, 48, 75],
        'x1:x2': [6, 24, 54, 96, 150],
        'x0:x1:x2': [6, 48, 162, 384, 750]
    }))

def test_linear_model_construct_design_matrix_covariates(linear_model: LinearModel) -> None:
    source = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],
        'C': [3, 6, 9, 12, 15],
        'Target': [10, 20, 30, 40, 50]
    })
    target = 'Target'
    features = ['A', 'B', 'C']
    covariates = ['D', 'E']
    alpha = 0.05
    linear_model = LinearModel(source, target, features, covariates, alpha)
    linear_model.construct_design_matrix()
    assert_frame_equal(linear_model.dmatrix, pd.DataFrame({
        'y': [10, 20, 30, 40, 50],
        'x0': [1, 2, 3, 4, 5],
        'x1': [2, 4, 6, 8, 10],
        'x2': [3, 6, 9, 12, 15],
        'e0': [0, 0, 0, 0, 0],
        'e1': [0, 0, 0, 0, 0]
    }))

def test_linear_model_construct_design_matrix_empty(linear_model):
    source = pd.DataFrame({
        'Target': [10, 20, 30, 40, 50]
    })
    target = 'Target'
    features = []
    covariates = []
    alpha = 0.05
    linear_model = LinearModel(source, target, features, covariates, alpha)
    linear_model.construct_design_matrix()
    assert_frame_equal(linear_model.dmatrix, pd.DataFrame({
        'y': [10, 20, 30, 40, 50]
    }))
