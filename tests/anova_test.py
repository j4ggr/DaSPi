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
from patsy.design_info import DesignInfo
from pandas.core.series import Series
from statsmodels.regression.linear_model import RegressionResultsWrapper

sys.path.append(str(Path(__file__).parent.resolve()))

import daspi

from daspi.anova import decode
from daspi.anova import optimize
from daspi.anova import hierarchical
from daspi.anova import get_term_name
from daspi.anova import is_main_feature
from daspi.anova import decode_cat_main
from daspi.anova import encoded_dmatrices
from daspi.anova import is_encoded_categorical
from daspi.anova import clean_categorical_names

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

    def test_get_term_name(self) -> None:
        assert get_term_name('A') == 'A'
        assert get_term_name('A[T.b]') == 'A'


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

    def test_basics(self) -> None:
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

        y, X_code, mapper, design_info = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code[X_code.columns], check_dtype=False)
        for key, value in expected_mapper.items():
            assert approx(mapper[key]) == value

    def test_empty(self) -> None:
        formula = 'Target ~ 1'
        expected_y = pd.DataFrame({'Target': [10, 20, 30, 40, 50, 60]})
        expected_X_code = pd.DataFrame({'Intercept': [1, 1, 1, 1, 1, 1]})
        expected_mapper = {}
        y, X_code, mapper, design_info = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code, check_dtype=False)
        assert mapper == expected_mapper
        assert isinstance(design_info, DesignInfo)

    def test_single_feature(self) -> None:
        formula = 'Target ~ A'
        expected_y = pd.DataFrame({'Target': [10, 20, 30, 40, 50, 60]})
        expected_X_code = pd.DataFrame({
            'Intercept': [1, 1, 1, 1, 1, 1],
            'A_b': [0, 1, 0, 0, 1, 0],
            'A_c': [0, 0, 1, 0, 0, 1]})
        
        expected_mapper = {
            'A_b': {'a | c': 0, 'b': 1},
            'A_c': {'a | b': 0, 'c': 1}}
        
        y, X_code, mapper, design_info = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code, check_dtype=False)
        assert mapper == expected_mapper
        assert isinstance(design_info, DesignInfo)

    def test_no_categorical(self) -> None:
        formula = 'Target ~ B + D'
        expected_y = pd.DataFrame({'Target': [10, 20, 30, 40, 50, 60]})
        expected_X_code = pd.DataFrame({
            'Intercept': [1, 1, 1, 1, 1, 1],
            'B': [-1, 0, 1, -1, 0, 1],
            'D': [-1, -0.6, -0.2, 0.2, 0.6, 1]})

        expected_mapper = {
            'B': {1.0: -1.0, 2.0: 0.0, 3.0: 1.0},
            'D': {1.0: -1.0, 2.0: -0.6, 3.0: -0.2, 4.0: 0.2, 5.0: 0.6, 6.0: 1.0}}

        y, X_code, mapper, design_info = encoded_dmatrices(self.data, formula)
        assert_frame_equal(y, expected_y, check_dtype=False)
        assert_frame_equal(X_code, expected_X_code, check_dtype=False)
        for key, value in expected_mapper.items():
            assert approx(mapper[key]) == value
        assert isinstance(design_info, DesignInfo)


class TestCleanCategoricalNames:

    def test_general(self) -> None:
        assert clean_categorical_names('A') == 'A'
        assert clean_categorical_names('A[T.b]') == 'A_b'
        assert clean_categorical_names('A[T.b]:C[T.d]') == 'A_b:C_d'
        assert clean_categorical_names('A[T.b]:C[T.d]:D') == 'A_b:C_d:D'
        assert clean_categorical_names('A[T.b]:C[T.d]:D[T.e]') == 'A_b:C_d:D_e'
        assert clean_categorical_names('A[T.b]:C') == 'A_b:C'
        assert clean_categorical_names('A[T.b]:C[T.d]:D[T.e]:E') == 'A_b:C_d:D_e:E'

    def test_no_encoding(self) -> None:
        assert clean_categorical_names('A') == 'A'
        assert clean_categorical_names('A:B') == 'A:B'
        assert clean_categorical_names('A:B:C') == 'A:B:C'

    def test_empty(self) -> None:
        assert clean_categorical_names('') == ''

    def test_whitespace(self) -> None:
        assert clean_categorical_names(' ') == ' '
        assert clean_categorical_names('  ') == '  '

    def test_no_match(self) -> None:
        assert clean_categorical_names('A[T]') == 'A[T]'
        assert clean_categorical_names('A[T.b') == 'A[T.b'
        assert clean_categorical_names('A[T.b]:C[T') == 'A_b:C[T'



@pytest.fixture
def lm() -> LinearModel:
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
    return LinearModel(source, target, features, covariates, alpha)

@pytest.fixture
def lm2() -> LinearModel:
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
    return LinearModel(source, target, features, covariates, alpha)

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
        assert_frame_equal(lm.source, pd.DataFrame({
            'A': [1, 0, 1, 1, 0, 1],
            'B': [-1, 1, 0, -1, 1, 0],
            'C': [3.1, 6.1, 9.1, 12.0, 15.0, 18.1],
            'bad': [0, 0, 0, 0, 0, 0],
            'Target': [11, 19, 30, 42, 49, 50]}))
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
        assert lm.dm_endogenous not in lm.dm_exogenous

    def test_model_property(self, lm: LinearModel) -> None:
        with pytest.raises(AssertionError):
            lm.model
        lm.fit()
        assert isinstance(lm.model, RegressionResultsWrapper)

    def test_least_feature(self, lm: LinearModel) -> None:
        lm.fit()
        assert any(lm.p_values.isna())
        assert lm.p_least > 0.05
        assert lm.least_feature() == lm.input_map['bad']
        assert lm._least_by_effect_() == lm.input_map['bad']
        assert lm._least_by_pvalue_() != lm.input_map['bad']

    def test_main_features_property(self, lm: LinearModel) -> None:
        lm.construct_design_matrix(complete=True)
        assert lm.main_features == ['x0', 'x1', 'x2', 'x3']
        lm.recursive_feature_elimination()
        assert lm.main_features == ['x2']

    def test_alpha_property(self, lm: LinearModel) -> None:
        assert lm.alpha == 0.05
        lm.alpha = 0.1
        assert lm.alpha == 0.1
        with pytest.raises(AssertionError):
            lm.alpha = -0.1
        with pytest.raises(AssertionError):
            lm.alpha = 1.1

    def test_dm_endogenous_property(self, lm: LinearModel) -> None:
        assert lm.dm_endogenous == 'y'

    def test_dm_exogenous_property(self, lm: LinearModel) -> None:
        lm.construct_design_matrix()
        lm.exclude = {'x0'}
        assert lm.dm_exogenous == ['Intercept', 'x1', 'x2', 'x3']

    def test_endogenous_property(self, lm: LinearModel) -> None:
        assert lm.endogenous == 'Target'

    def test_exogenous_property(self, lm: LinearModel) -> None:
        lm.construct_design_matrix()
        lm.exclude = {'x0'}
        assert lm.exogenous == ['Intercept', 'B', 'C', 'bad']
        
        lm.construct_design_matrix(complete=True)
        lm.exclude = {c for c in lm.dmatrix.columns if 'x3' in c}
        expected_exogenous = [
            'Intercept',
            'A',
            'B',
            'A:B',
            'C',
            'A:C',
            'B:C',
            'A:B:C']
        assert lm.exogenous == expected_exogenous

    def test_construct_design_matrix_no_encode_no_complete(self, lm2: LinearModel) -> None:
        lm2.construct_design_matrix()
        assert_frame_equal(lm2.dmatrix, pd.DataFrame({
            'y': [10, 20, 30, 40, 50, 60],
            'Intercept': [1, 1, 1, 1, 1, 1],
            'x0_b': [0, 1, 0, 0, 1, 0],
            'x0_c': [0, 0, 1, 0, 0, 1],
            'x2_True': [1, 0, 1, 0, 1, 0],
            'x1': [1, 2, 3, 1, 2, 3],
            'e0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), check_dtype=False)

    def test_construct_design_matrix_no_encode_complete(
            self, lm2: LinearModel) -> None:
        lm2.construct_design_matrix(encode=False, complete=True)
        assert_frame_equal(lm2.dmatrix, pd.DataFrame({
            'y': [10, 20, 30, 40, 50, 60],
            'Intercept': [1, 1, 1, 1, 1, 1],
            'x0_b': [0, 1, 0, 0, 1, 0],
            'x0_c': [0, 0, 1, 0, 0, 1],
            'x2_True': [1, 0, 1, 0, 1, 0],
            'x0_b:x2_True': [0, 0, 0, 0, 1, 0],
            'x0_c:x2_True': [0, 0, 1, 0, 0, 0],
            'x1': [1, 2, 3, 1, 2, 3],
            'x0_b:x1': [0, 2, 0, 0, 2, 0],
            'x0_c:x1': [0, 0, 3, 0, 0, 3],
            'x1:x2_True': [1, 0, 3, 0, 2, 0],
            'x0_b:x1:x2_True': [0, 0, 0, 0, 2, 0],
            'x0_c:x1:x2_True': [0, 0, 3, 0, 0, 0],
            'e0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), check_dtype=False)

    def test_construct_design_matrix_encode_no_complete(
            self, lm2: LinearModel) -> None:
        lm2.construct_design_matrix(encode=True)
        assert_frame_equal(lm2.dmatrix, pd.DataFrame({
            'y': [10, 20, 30, 40, 50, 60],
            'Intercept': [1, 1, 1, 1, 1, 1],
            'x0_b': [0, 1, 0, 0, 1, 0],
            'x0_c': [0, 0, 1, 0, 0, 1],
            'x2_True': [1, 0, 1, 0, 1, 0],
            'x1': [-1, 0, 1, -1, 0, 1],
            'e0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), check_dtype=False)

    def test_construct_design_matrix_encode_complete(
            self, lm2: LinearModel) -> None:
        lm2.construct_design_matrix(encode=True, complete=True)
        assert_frame_equal(lm2.dmatrix, pd.DataFrame({
            'y': [10, 20, 30, 40, 50, 60],
            'Intercept': [1, 1, 1, 1, 1, 1],
            'x0_b': [0, 1, 0, 0, 1, 0],
            'x0_c': [0, 0, 1, 0, 0, 1],
            'x2_True': [1, 0, 1, 0, 1, 0],
            'x0_b:x2_True': [0, 0, 0, 0, 1, 0],
            'x0_c:x2_True': [0, 0, 1, 0, 0, 0],
            'x1': [-1, 0, 1, -1, 0, 1],
            'x0_b:x1': [0, 0, 0, 0, 0, 0],
            'x0_c:x1': [0, 0, 1, 0, 0, 1],
            'x1:x2_True': [-1, 0, 1, 0, 0, 0],
            'x0_b:x1:x2_True': [0, 0, 0, 0, 0, 0],
            'x0_c:x1:x2_True': [0, 0, 1, 0, 0, 0],
            'e0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), check_dtype=False)
        
    def test_construct_design_matrix_covariates(self, lm: LinearModel) -> None:
        source = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [3, 6, 9, 12, 15],
            'D': [3.1, 6.1, 9.1, 12.0, 15.0],
            'Target': [10, 20, 30, 40, 50]
        })
        target = 'Target'
        features = ['A', 'B', 'C']
        covariates = ['D']
        alpha = 0.05
        lm = LinearModel(source, target, features, covariates, alpha)
        lm.construct_design_matrix()
        assert_frame_equal(lm.dmatrix, pd.DataFrame({
            'y': [10, 20, 30, 40, 50],
            'Intercept': [1.0, 1.0, 1.0, 1.0, 1.0],
            'x0': [1, 2, 3, 4, 5],
            'x1': [2, 4, 6, 8, 10],
            'x2': [3, 6, 9, 12, 15],
            'e0': [3.1, 6.1, 9.1, 12.0, 15.0]}), check_dtype=False)
        lm.construct_design_matrix(complete=True)
        assert any([(':' in c) for c in lm.dm_exogenous])
        assert not any([(':e0' in c) for c in lm.dm_exogenous])
        assert not any([('e0:' in c) for c in lm.dm_exogenous])
    
    def test_anova(self, anova3_s_valid: DataFrame, anova3_c_valid: DataFrame) -> None:
        df = daspi.load_dataset('anova3')
        lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'])
        
        anova = lm.construct_design_matrix().fit().anova()
        valid = anova3_s_valid
        assert_series_equal(
            anova['df'], valid['df'], check_dtype=False)
        s = str(anova)
        assert_series_equal(
            anova['sum_sq'], valid['sum_sq'], check_exact=False, atol=1e-2)

        anova = lm.construct_design_matrix(complete=True).fit().anova()
        valid = anova3_c_valid
        assert_series_equal(
            anova['df'], valid['df'], check_dtype=False)
        assert_series_equal(
            anova['sum_sq'], valid['sum_sq'], check_exact=False, atol=1e-2)

