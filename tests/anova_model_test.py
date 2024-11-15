import sys
import patsy.design_info
import patsy.highlevel
import pytest

import pandas as pd

from pytest import approx
from pathlib import Path
from pandas.testing import assert_frame_equal
from pandas.core.frame import DataFrame
from statsmodels.regression.linear_model import RegressionResultsWrapper

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi import load_dataset
from daspi.anova.model import *


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



@pytest.fixture
def lm() -> LinearModel:
    """LinearModel no continuous, simple"""
    source = pd.DataFrame({
        'A': [1, 0, 1, 1, 0, 1],
        'B': [-1, 1, 0, -1, 1, 0],
        'C': [3.1, 6.1, 9.1, 12.0, 15.0, 18.1],
        'bad': [0, 0, 0, 0, 0, 0],
        'Target': [11, 19, 30, 42, 49, 50]})
    target = 'Target'
    categorical = ['A', 'B', 'C', 'bad']
    continuous = []
    alpha = 0.05
    return LinearModel(
        source, target, categorical, continuous, alpha,
        order=1, encode_categoricals=False)

@pytest.fixture
def lm2() -> LinearModel:
    """LinearModel no continuous, complete"""
    source = pd.DataFrame({
        'A': [1, 0, 1, 1, 0, 1],
        'B': [-1, 1, 0, -1, 1, 0],
        'C': [3.1, 6.1, 9.1, 12.0, 15.0, 18.1],
        'bad': [0, 0, 0, 0, 0, 0],
        'Target': [11, 19, 30, 42, 49, 50]})
    target = 'Target'
    categorical = ['A', 'B', 'C', 'bad']
    continuous = []
    alpha = 0.05
    return LinearModel(
        source, target, categorical, continuous, alpha,
        order=4, encode_categoricals=False, skip_intercept_as_least=True)

@pytest.fixture
def lm3() -> LinearModel:
    """LinearModel with continuous, simple"""
    source = pd.DataFrame({
        'A': ['a', 'b', 'c', 'a', 'b', 'c'],
        'B': [1, 2, 3, 1, 2, 3],
        'C': [True, False, True, False, True, False],
        'D': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Target': [10, 20, 30, 40, 50, 60]})
    target = 'Target'
    categorical = ['A', 'B', 'C']
    continuous = ['D']
    alpha = 0.05
    return LinearModel(
        source, target, categorical, continuous, alpha,
        order=1, encode_categoricals=False)

@pytest.fixture
def lm4() -> LinearModel:
    """LinearModel with continuous, complete"""
    source = pd.DataFrame({
        'A': ['a', 'b', 'c', 'a', 'b', 'c'],
        'B': [1, 2, 3, 1, 2, 3],
        'C': [True, False, True, False, True, False],
        'D': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Target': [10, 20, 30, 40, 50, 60]})
    target = 'Target'
    categorical = ['A', 'B', 'C']
    continuous = ['D']
    alpha = 0.05
    return LinearModel(
        source, target, categorical, continuous, alpha,
        order=3, encode_categoricals=False)

@pytest.fixture
def anova3_c_valid() -> DataFrame:
    df = pd.read_csv(
        valid_data_dir/'anova3_result.csv', skiprows=8, sep=';', index_col=0)
    return df

@pytest.fixture
def anova3_s_valid() -> DataFrame:
    df = pd.read_csv(
        valid_data_dir/'anova3_result.csv', skiprows=1, skipfooter=11, sep=';',
        index_col=0)
    return df


class TestLinearModel:

    def test_init(self, lm: LinearModel) -> None:
        assert_frame_equal(lm.data, pd.DataFrame({
            'x0': [1, 0, 1, 1, 0, 1],
            'x1': [-1, 1, 0, -1, 1, 0],
            'x2': [3.1, 6.1, 9.1, 12.0, 15.0, 18.1],
            'x3': [0, 0, 0, 0, 0, 0],
            'y': [11, 19, 30, 42, 49, 50]}), check_dtype=False)
        assert lm.target == 'Target'
        assert lm.categorical == ['A', 'B', 'C', 'bad']
        assert lm.continuous == []
        assert lm.alpha == 0.05
        assert lm.output_map == {'Target': 'y'}
        assert lm.input_map == {'A': 'x0', 'B': 'x1', 'C': 'x2', 'bad': 'x3'}
        assert lm.excluded == set()
        assert lm._model is None
    
    def test_formula(self, lm: LinearModel, lm2: LinearModel) -> None:
        assert '*' not in lm.formula
        assert '+' in lm.formula
        assert ':' not in lm.formula
        lm.fit()
        assert '*' not in lm.formula
        assert '+' in lm.formula
        assert ':' not in lm.formula

        assert '*' not in lm2.formula
        assert '+' in lm2.formula
        assert ':' in lm2.formula
        lm2.fit()
        assert '*' not in lm2.formula
        assert '+' in lm2.formula
        assert ':' in lm2.formula
        lm2.excluded.add('Intercept')
        assert '-1' in lm2.formula

    def test_model_property(self, lm: LinearModel) -> None:
        with pytest.raises(AssertionError):
            lm.model
        lm.fit()
        assert isinstance(lm.model, RegressionResultsWrapper)

    def test_least_term(self, lm: LinearModel, lm4: LinearModel) -> None:
        lm4.fit()
        assert all(lm4.p_values().isna())
        assert lm4.least_term() == 'A:B:C'

        p_values = lm.fit().p_values()
        least = lm.least_term()
        assert any(p_values.isna())
        assert p_values.max() > 0.05
        assert p_values.idxmax() != least
        assert least == 'bad'

        p_values = lm.eliminate('bad').fit().p_values()
        assert not any(p_values.isna())
        assert p_values.max() > 0.05
        assert lm.least_term() == 'A'

    def test_main_features_property(self, lm2: LinearModel) -> None:
        lm2.fit()
        assert lm2.main_features == ['x0', 'x1', 'x2', 'x3']
        formula = ''
        gof = pd.DataFrame()
        for i, _gof in enumerate(lm2.recursive_feature_elimination()):
            assert _gof.loc[i, 'formula'] != formula
            formula = _gof.loc[i, 'formula']
            gof = pd.concat([gof, _gof])
        assert gof['p_least'].iloc[-2] >= lm2.alpha
        assert lm2.main_features == ['x2']

    def test_alpha_property(self, lm: LinearModel) -> None:
        assert lm.alpha == 0.05
        lm.alpha = 0.1
        assert lm.alpha == 0.1
        with pytest.raises(AssertionError):
            lm.alpha = -0.1
        with pytest.raises(AssertionError):
            lm.alpha = 1.1
    
    def test_anova(self, anova3_s_valid: DataFrame, anova3_c_valid: DataFrame) -> None:
        df = load_dataset('anova3')
        lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'])
        
        valid = anova3_s_valid
        anova = lm.fit().anova('III')
        anova.columns.name = None
        anova = anova.loc[valid.index.to_list(), valid.columns.to_list()]
        assert_frame_equal(
            anova, valid, check_dtype=False, check_exact=False,
            atol=1e-2)

        valid = anova3_c_valid
        lm = LinearModel(
            df, 'Cholesterol', ['Sex', 'Risk', 'Drug'], order=3)
        anova = lm.fit().anova('III')
        anova.columns.name = None
        anova = anova.loc[valid.index.to_list(), valid.columns.to_list()]
        assert_frame_equal(
            anova[valid.columns], valid, check_dtype=False, check_exact=False,
            atol=1e-2)

    def test_summary(self) -> None:
        df = load_dataset('anova3')
        lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug']).fit()
        lm2 = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'], order=3).fit()
        expected_param_table = (
            '===============================================================================\n'
            '                  coef    std err          t      P>|t|      [0.025      0.975]\n'
            '-------------------------------------------------------------------------------\n'
            'Intercept       5.7072      0.268     21.295      0.000       5.170       6.244\n'
            'Sex[T.M]        0.3719      0.240      1.551      0.127      -0.109       0.852\n'
            'Risk[T.Low]    -0.8692      0.240     -3.626      0.001      -1.350      -0.389\n'
            'Drug[T.B]      -0.1080      0.294     -0.368      0.714      -0.696       0.480\n'
            'Drug[T.C]       0.1750      0.294      0.596      0.554      -0.413       0.763\n'
            '===============================================================================')
        smry = lm.summary()
        assert len(smry.tables) == 3
        assert smry.tables[-1].as_text() == expected_param_table

        expected_anova_table = (
            '============================================================================\n'
            '                   DF         SS         MS          F          p         n2\n'
            '----------------------------------------------------------------------------\n'
            'Intercept           1    390.868    390.868    453.467    3.1e-28      0.864\n'
            'Sex                 1      2.075      2.075      2.407      0.127      0.005\n'
            'Risk                1     11.332     11.332     13.147      0.001      0.025\n'
            'Drug                2      0.816      0.408      0.473      0.626      0.002\n'
            'Residual           55     47.407      0.862        nan        nan      0.105\n'
            '============================================================================')
        smry_anova = lm.summary(anova_typ='III', vif=False)
        assert len(smry_anova.tables) == 4
        assert 'ANOVA Typ-III' in smry_anova.tables[0].title
        assert smry_anova.tables[-1].as_text() == expected_anova_table

        expected_anova_table = (
            '=======================================================================================\n'
            '                   DF         SS         MS          F          p         n2        VIF\n'
            '---------------------------------------------------------------------------------------\n'
            'Sex                 1      2.075      2.075      2.407      0.127      0.034      1.000\n'
            'Risk                1     11.332     11.332     13.147      0.001      0.184      1.000\n'
            'Drug                2      0.816      0.408      0.473      0.626      0.013      1.000\n'
            'Residual           55     47.407      0.862        nan        nan      0.769        nan\n'
            '=======================================================================================')
        smry_vif_anova = lm.summary(anova_typ='', vif=True)
        assert len(smry_vif_anova.tables) == 4
        assert 'ANOVA Typ-II' in smry_vif_anova.tables[0].title
        assert smry_vif_anova.tables[-1].as_text() == expected_anova_table

        expected_anova_table = (
            '==========================================================================================\n'
            '                      DF         SS         MS          F          p         n2        VIF\n'
            '------------------------------------------------------------------------------------------\n'
            'Sex                    1      2.075      2.075      2.462      0.123      0.034      1.000\n'
            'Risk                   1     11.332     11.332     13.449      0.001      0.184      1.000\n'
            'Drug                   2      0.816      0.408      0.484      0.619      0.013      1.000\n'
            'Sex:Risk               1      0.117      0.117      0.139      0.711      0.002      1.364\n'
            'Sex:Drug               2      2.564      1.282      1.522      0.229      0.042      1.616\n'
            'Risk:Drug              2      2.438      1.219      1.446      0.245      0.040      1.616\n'
            'Sex:Risk:Drug          2      1.844      0.922      1.094      0.343      0.030      1.000\n'
            'Residual              48     40.445      0.843        nan        nan      0.656        nan\n'
            '==========================================================================================')
        smry2 = lm2.summary(anova_typ='', vif=True)
        assert len(smry2.tables) == 4
        assert 'ANOVA Typ-II' in smry2.tables[0].title
        assert smry2.tables[-1].as_text() == expected_anova_table
    
    def test_r2_pred(self) -> None:
        """Source:
        https://www.additive-net.de/de/software/support/minitab-support/minitab-faq-analysen/3498-minitab-r19-doe-r-qd-prog"""
        data = pd.DataFrame({
            'Ergebnis': [7.5292, 11.1151, 8.3440, 11.9081, 10.9183, 11.7622],
            'A': [-1, 1, 1, -1, -1, 1],
            'B': [0, -1, 0, -1, 1, 1],
            'Center': [1, 0, 1, 0, 0, 0]})
        lm = LinearModel(
                data, 'Ergebnis', ['A', 'B'], ['Center'], order=2, 
                encode_categoricals=False
            ).fit()
        assert lm.r2_pred(), approx(0.350426519634100657)
    
    def test_str(self) -> None:
        df = load_dataset('anova3')
        lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug']).fit()
        expected_str = 'Cholesterol ~ 5.7072 + 0.3719*Sex[T.M] - 0.8692*Risk[T.Low] - 0.1080*Drug[T.B] + 0.1750*Drug[T.C]'
        assert str(lm) == expected_str

    def test_eliminate_term(self, lm3: LinearModel) -> None:
        lm3.fit()
        lm3.eliminate('A')
        assert 'x0' in lm3.excluded

        lm3.eliminate('B')
        assert 'x1' in lm3.excluded

        lm3.eliminate('C')
        assert 'x2' in lm3.excluded

    def test_eliminate_interaction_term(self, lm4: LinearModel) -> None:
        lm4.fit()
        lm4.eliminate('A:B')
        assert 'x0:x1' in lm4.excluded

        lm4.eliminate('A:C')
        assert 'x0:x2' in lm4.excluded

        lm4.eliminate('B:C')
        assert 'x1:x2' in lm4.excluded

    def test_eliminate_invalid_term(self, lm3: LinearModel) -> None:
        lm3.fit()
        with pytest.raises(AssertionError, match=r'Given term Q is not in model'):
            lm3.eliminate('Q')

        with pytest.raises(AssertionError, match=r'Given term x0:x1:x2 is not in model'):
            lm3.eliminate('A:B:C')

    def test_eliminate_encoded_term(self, lm3: LinearModel) -> None:
        lm3.fit()
        lm3.eliminate('x0')
        assert 'x0' in lm3.excluded

        lm3.eliminate('x1[T.b]')
        assert 'x1' in lm3.excluded

        lm3.eliminate('x2[T.2.2]')
        assert 'x2' in lm3.excluded
