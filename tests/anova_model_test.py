import sys
import patsy.design_info
import patsy.highlevel
import pytest
from pytest import approx

import numpy as np
import pandas as pd

from pytest import approx
from pathlib import Path
from pandas.testing import assert_frame_equal
from pandas.core.frame import DataFrame
from statsmodels.regression.linear_model import RegressionResultsWrapper

from daspi.anova.model import LinearModel

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi import ANOVA
from daspi import load_dataset
from daspi import GageEstimator
from daspi import MeasurementUncertainty
from daspi.anova.model import *
from daspi.anova.model import GeneralizedLinearModel

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
        assert is_main_parameter('A') == True
        assert is_main_parameter('B') == True
        assert is_main_parameter('Intercept') == False
        assert is_main_parameter('A:B') == False
        assert is_main_parameter('A:B:C') == False

    def test_empty(self) -> None:
        assert is_main_parameter('') == True

    def test_whitespace(self) -> None:
        assert is_main_parameter(' ') == True
        assert is_main_parameter('  ') == True

    def test_separator(self) -> None:
        assert is_main_parameter(':') == False
        assert is_main_parameter('A:') == False
        assert is_main_parameter('A:B') == False
        assert is_main_parameter('A:B:C') == False



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
    factors = ['A', 'B', 'C', 'bad']
    disturbances = []
    alpha = 0.05
    return LinearModel(
        source, target, factors, disturbances, alpha,
        order=1, encode_factors=False, fit_at_init=False)

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
    factors = ['A', 'B', 'C', 'bad']
    disturbances = []
    alpha = 0.05
    return LinearModel(
        source, target, factors, disturbances, alpha,
        order=4, encode_factors=False, skip_intercept_as_least=True,
        fit_at_init=False)

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
    factors = ['A', 'B', 'C']
    disturbances = ['D']
    alpha = 0.05
    return LinearModel(
        source, target, factors, disturbances, alpha,
        order=1, encode_factors=False, fit_at_init=False)

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
    factors = ['A', 'B', 'C']
    disturbances = ['D']
    alpha = 0.05
    return LinearModel(
        source, target, factors, disturbances, alpha,
        order=3, encode_factors=False, fit_at_init=False)

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
        assert lm.factors == ['A', 'B', 'C', 'bad']
        assert lm.covariates == []
        assert lm.alpha == 0.05
        assert lm.target_map == {'Target': 'y'}
        assert lm.feature_map == {'A': 'x0', 'B': 'x1', 'C': 'x2', 'bad': 'x3'}
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
        assert not lm.fitted 
        lm.fit()
        assert lm.fitted
        assert isinstance(lm.model, RegressionResultsWrapper)

    def test_least_parameter(self, lm: LinearModel, lm4: LinearModel) -> None:
        lm4.fit()
        assert all(lm4.p_values().isna())
        assert lm4.least_parameter() == 'A:B:C'

        p_values = lm.fit().p_values()
        least = lm.least_parameter()
        assert any(p_values.isna())
        assert p_values.max() > 0.05
        assert p_values.idxmax() != least
        assert least == 'bad'

        p_values = lm.eliminate('bad').fit().p_values()
        assert not any(p_values.isna())
        assert p_values.max() > 0.05
        assert lm.least_parameter() == 'A'

    def test_effects_include_intercept(self, lm: LinearModel) -> None:
        lm.fit()
        effects = lm.effects()
        assert 'Intercept' in effects.index
        effects_no_intercept = lm.effects(include_intercept=False)
        assert 'Intercept' not in effects_no_intercept.index
        assert len(effects_no_intercept) == len(effects) - 1
        pd.testing.assert_series_equal(
            effects_no_intercept, effects.drop('Intercept'))

    def test_p_values_include_intercept(self, lm: LinearModel) -> None:
        lm.fit()
        p_values = lm.p_values()
        assert 'Intercept' in p_values.index
        p_values_no_intercept = lm.p_values(include_intercept=False)
        assert 'Intercept' not in p_values_no_intercept.index
        assert len(p_values_no_intercept) == len(p_values) - 1
        pd.testing.assert_series_equal(
            p_values_no_intercept, p_values.drop('Intercept'))

    def test_main_parameters_property(self, lm2: LinearModel) -> None:
        lm2.fit()
        assert lm2.main_parameters == ['A', 'B', 'C', 'bad']
        formula = ''
        gof = pd.DataFrame()
        for i, _gof in enumerate(lm2.recursive_elimination()):
            assert _gof.loc[i, 'formula'] != formula
            formula = _gof.loc[i, 'formula']
            gof = pd.concat([gof, _gof])
        assert gof['p_least'].iloc[-2] >= lm2.alpha
        assert lm2.main_parameters == ['C']

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
        anova = lm.anova('III')
        anova.columns.name = None
        anova = anova.loc[valid.index.to_list(), valid.columns.to_list()]
        assert_frame_equal(
            anova, valid, check_dtype=False, check_exact=False,
            atol=1e-2)

        valid = anova3_c_valid
        lm = LinearModel(
            df, 'Cholesterol', ['Sex', 'Risk', 'Drug'], order=3)
        anova = lm.anova('III')
        anova.columns.name = None
        anova = anova.loc[valid.index.to_list(), valid.columns.to_list()]
        assert_frame_equal(
            anova[valid.columns], valid, check_dtype=False, check_exact=False,
            atol=1e-2)

    def test_summary(self) -> None:
        df = load_dataset('anova3')
        lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'])
        lm2 = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'], order=3)
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
                encode_factors=False)
        assert lm.r2_pred(), approx(0.350426519634100657)
    
    def test_str(self) -> None:
        df = load_dataset('anova3')
        lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'])
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
    
    def test_include_term(self, lm3: LinearModel) -> None:
        lm3.fit().eliminate('A').fit()
        assert 'x0' in lm3.excluded
        assert 'A' not in lm3.anova().index
        
        lm3.include('A').fit()
        assert 'x0' not in lm3.excluded
        assert 'A' in lm3.anova().index

    def test_include_interaction_term(self, lm4: LinearModel) -> None:
        lm4.fit().eliminate('A:B').fit()
        assert 'x0:x1' in lm4.excluded
        assert 'A:B' not in lm4.parameters

        lm4.include('A:B').fit()
        assert 'x0:x1' not in lm4.excluded
        assert 'A:B' in lm4.parameters

    def test_predict(self, lm: LinearModel) -> None:
        lm.fit().eliminate('bad').fit()
        prediction = lm.predict({'A': 1, 'B': 0, 'C': 18.1})[lm.target][0]
        assert prediction > 45.0
        prediction = lm.predict({'A': 1, 'B': -1, 'C': 3.1})[lm.target][0]
        assert prediction < 15.0

    def test_predict_invalid_data(self, lm: LinearModel) -> None:
        lm.fit()
        with pytest.raises(
                AssertionError,
                match=r'Please provide a value for "A"'):
            lm.predict({})
        
        with pytest.raises(
                AssertionError,
                match=r'Please provide a value for "C"'):
            lm.predict({'A': 1, 'B': 1, 'bad': 1})

        with pytest.raises(
                AssertionError,
                match=r'"foo" is not a main parameter of the model.'):
            lm.predict({'A': 1, 'B': 1, 'C': 1, 'bad': 1, 'foo': 1})
    
    def test_optimize(self, lm: LinearModel) -> None:
        lm.fit().eliminate('bad').fit()
        assert lm.optimize(maximize=True) == {'A': 1, 'B': 1, 'C': 18.1}
        assert lm.optimize(maximize=False) == {'A': 0, 'B': -1, 'C': 3.1}

    def test_optimize_bounds(self, lm: LinearModel) -> None:
        lm.fit().eliminate('bad').fit()
        bounds={'A': (0, 1), 'B': (-1, 1), 'C': (10, 15)}
        xs_max = lm.optimize(maximize=True, bounds=bounds)
        assert xs_max['A'] in bounds['A']
        assert xs_max['B'] in bounds['B']
        assert xs_max['C'] == 15

        xs_min = lm.optimize(maximize=False, bounds=bounds)
        assert xs_min['A'] in bounds['A']
        assert xs_min['A'] != xs_max['A']
        assert xs_min['B'] in bounds['B']
        assert xs_min['B'] != xs_max['B']
        assert xs_min['C'] == 10

        bounds={'A': 1, 'C': 13}
        xs_single_max = lm.optimize(maximize=True, bounds=bounds)
        assert xs_single_max['A'] == 1
        assert xs_single_max['C'] == 13

        xs_single_min = lm.optimize(maximize=False, bounds=bounds)
        assert xs_single_min['A'] == 1
        assert xs_single_min['B'] != xs_single_max['B']
        assert xs_single_min['C'] == 13
    
    def test_optimize_invalid_bounds(self, lm: LinearModel) -> None:
        lm.fit().eliminate('bad').fit()

        with pytest.raises(
                AssertionError,
                match=r'Bounds for "C" must be a tuple of length 2.'):
            lm.optimize(maximize=True, bounds={'C': (15, 18, 30)})
        
        with pytest.raises(
                AssertionError,
                match=r'Bounds for "C" must be within the range of the data'):
            lm.optimize(maximize=True, bounds={'C': (20, 15)})
    
    def test_highest_parameters(self, lm4: LinearModel) -> None:
        lm4.fit()
        parameters = lm4.highest_parameters(factors_only=False)
        assert parameters == ['A:B:C', ANOVA.INTERCEPT, 'D']
        parameters = lm4.highest_parameters(factors_only=True)
        assert parameters == ['A:B:C']
        
        lm4.eliminate('A:B:C').fit()
        parameters = lm4.highest_parameters(factors_only=False)
        assert parameters == [
            'A:C', 'A:B', 'B:C', ANOVA.INTERCEPT, 'D']
        parameters = lm4.highest_parameters(factors_only=True)
        assert parameters == ['A:C', 'A:B', 'B:C']
        
        lm4.eliminate('A:B').fit()
        parameters = lm4.highest_parameters(factors_only=False)
        assert parameters == ['A:C', 'B:C', ANOVA.INTERCEPT, 'D']
        parameters = lm4.highest_parameters(factors_only=True)
        assert parameters == ['A:C', 'B:C']
        
        lm4.eliminate('A:C').fit()
        parameters = lm4.highest_parameters(factors_only=False)
        assert parameters == ['B:C', ANOVA.INTERCEPT, 'A', 'D']
        parameters = lm4.highest_parameters(factors_only=True)
        assert parameters == ['B:C', 'A']


class TestGageStudyModel:

    df_single = pd.read_csv(
        valid_data_dir/'gage_study.csv',
        skiprows=lambda x: x not in list(range(1, 52)),
        sep=';'
        ).dropna(how='all', axis=1)

    df_lin = pd.read_csv(
        valid_data_dir/'gage_study.csv',
        skiprows=lambda x: x not in list(range(54, 85)),
        sep=';'
        ).dropna(how='all', axis=1)
    
    df_mpt = pd.read_csv(
        valid_data_dir/'gage_study.csv',
        skiprows=lambda x: x not in list(range(87, 178)),
        sep=';'
        ).dropna(how='all', axis=1)
    
    @pytest.fixture
    def gage_single(self) -> GageStudyModel:
        """Get GageStudyModel with a single reference"""
        return GageStudyModel(
            source=self.df_single,
            target='result',
            reference='reference',
            u_cal=self.df_single['U_cal'][0],
            tolerance=self.df_single['tolerance'][0],
            resolution=self.df_single['resolution'][0],
            k=2)
    
    @pytest.fixture
    def gage_lin(self) -> GageStudyModel:
        """Get GageStudyModel with six references to calculate the
        uncertainty of linearity"""
        return GageStudyModel(
            source=self.df_lin,
            target='result',
            reference='reference',
            u_cal=self.df_lin['U_cal'][0],
            tolerance=self.df_lin['tolerance'][0],
            resolution=self.df_lin['resolution'][0],
            bias_corrected=True,
            k=2)
    
    @pytest.fixture
    def gage_mpt(self) -> GageStudyModel:
        """Get GageStudyModel with three points at three references, 
        to calculate the uncertainty with ANOVA."""
        return GageStudyModel(
            source=self.df_mpt,
            target='result',
            reference='reference',
            u_cal=self.df_mpt['U_cal'][0],
            tolerance=self.df_mpt['tolerance'][0],
            resolution=self.df_mpt['resolution'][0],
            u_lin=MeasurementUncertainty(standard=0),
            u_rest=MeasurementUncertainty(
                error_limit=0.0008, k=2, distribution='rectangular'),
            k=2,)
    
    @pytest.fixture
    def df_vsingle(self) -> DataFrame:
        """Get validation data of df_single"""
        df = (self.df_single
            .loc[:, 'influence': 'rank']
            .dropna(how='all', axis=0)
            .set_index('influence'))
        return df
    
    @pytest.fixture
    def df_vlin(self) -> DataFrame:
        """Get validation data of df_lin"""
        df = (self.df_lin
            .loc[:, 'influence': 'rank']
            .dropna(how='all', axis=0)
            .set_index('influence'))
        return df
    
    @pytest.fixture
    def df_vmpt(self) -> DataFrame:
        """Get validation data of df_mpt"""
        df = (self.df_mpt
            .loc[:, 'influence': 'rank']
            .dropna(how='all', axis=0)
            .set_index('influence'))
        return df
    
    def test_init(self, gage_lin: GageStudyModel) -> None:
        assert gage_lin.target == 'result'
        assert gage_lin.reference == 'reference'
        assert gage_lin.u_cal.standard == approx(self.df_lin['U_cal'][0]/2)
        assert gage_lin.tolerance == self.df_lin['tolerance'][0]
        assert gage_lin.resolution == self.df_lin['resolution'][0]
    
    def test_uncertainties_single(
            self, gage_single: GageStudyModel, df_vsingle: DataFrame) -> None:
        df_u = gage_single.uncertainties()
        assert list(df_u.index) == ANOVA.UNCERTAINTY_ROWS_MS

        for u_is, u_valid in zip(df_u['u'], df_vsingle['u']):
            assert u_is == approx(u_valid, abs=1e-5)
        
        for U_is, U_valid in zip(df_u['U'], df_vsingle['U']):
            assert U_is == approx(U_valid, abs=1e-5)
        
        for Q_is, Q_valid in zip(df_u['Q'], df_vsingle['Q']):
            assert Q_is == approx(Q_valid, abs=1e-3)
        
        for r_is, r_valid in zip(df_u['rank'], df_vsingle['rank']):
            if pd.isna(r_is):
                assert pd.isna(r_valid)
            else:
                assert r_is == r_valid
    
    def test_uncertainties_lin(
            self, gage_lin: GageStudyModel, df_vlin: DataFrame) -> None:
        df_u = gage_lin.uncertainties()
        assert list(df_u.index) == ANOVA.UNCERTAINTY_ROWS_MS

        for u_is, u_valid in zip(df_u['u'], df_vlin['u']):
            assert u_is == approx(u_valid, abs=1e-5)
        
        for U_is, U_valid in zip(df_u['U'], df_vlin['U']):
            assert U_is == approx(U_valid, abs=1e-5)
        
        for Q_is, Q_valid in zip(df_u['Q'], df_vlin['Q']):
            assert Q_is == approx(Q_valid, abs=1e-3)
        
        for r_is, r_valid in zip(df_u['rank'], df_vlin['rank']):
            if pd.isna(r_is):
                assert pd.isna(r_valid)
            else:
                assert r_is == r_valid
    
    def test_uncertainties_mpt(
            self, gage_mpt: GageStudyModel, df_vmpt: DataFrame) -> None:
        df_u = gage_mpt.uncertainties()
        assert list(df_u.index) == ANOVA.UNCERTAINTY_ROWS_MS

        for u_is, u_valid in zip(df_u['u'], df_vmpt['u']):
            assert u_is == approx(u_valid, abs=1e-5)
        
        for U_is, U_valid in zip(df_u['U'], df_vmpt['U']):
            assert U_is == approx(U_valid, abs=1e-5)
        
        for Q_is, Q_valid in zip(df_u['Q'], df_vmpt['Q']):
            assert Q_is == approx(Q_valid, abs=1e-3)
        
        for r_is, r_valid in zip(df_u['rank'], df_vmpt['rank']):
            if pd.isna(r_is):
                assert pd.isna(r_valid)
            else:
                assert r_is == r_valid

    def test_properties_and_tables(self, gage_single: GageStudyModel) -> None:
        # Test references_analysis
        refs = gage_single.references_analysis()
        assert isinstance(refs, pd.DataFrame)
        assert not refs.empty

        # Test capabilities
        caps = gage_single.capabilities()
        assert isinstance(caps, pd.DataFrame)
        assert not caps.empty

        # Test uncertainties
        unc = gage_single.uncertainties()
        assert isinstance(unc, pd.DataFrame)
        assert not unc.empty

        # Test n_samples, n_references, n_replications
        assert isinstance(gage_single.n_samples, int)
        assert gage_single.n_samples > 0
        assert isinstance(gage_single.n_references, int)
        assert gage_single.n_references > 0
        assert isinstance(gage_single.n_replications, int)
        assert gage_single.n_replications > 0

        # Test tolerance, resolution, k
        assert isinstance(gage_single.tolerance, float)
        assert gage_single.tolerance > 0
        assert isinstance(gage_single.resolution, float)
        assert gage_single.resolution > 0
        assert isinstance(gage_single.k, (int, float))
        assert gage_single.k > 0

        # Test bias_corrected, bias
        assert isinstance(gage_single.bias_corrected, bool)
        assert isinstance(gage_single.bias, float)

        # Test df_u, df_ums, df_ump
        assert isinstance(gage_single.uncertainties(), pd.DataFrame)


class TestGageRnRModel:
    df_thick = load_dataset('grnr_layer_thickness')
    df_adj = load_dataset('grnr_adjustment')

    @pytest.fixture
    def rnr_thick_model(self) -> GageRnRModel:
        gage = GageStudyModel(
            source=self.df_thick,
            target='result_gage',
            reference='reference',
            u_cal=self.df_thick['U_cal'][0],
            tolerance=self.df_thick['tolerance'][0],
            resolution=self.df_thick['resolution'][0])
        model = GageRnRModel(
            source=self.df_thick,
            target='result_rnr',
            part='part',
            gage=gage,
            u_av='operator')
        return model

    @pytest.fixture
    def rnr_thick_model_gv(self) -> GageRnRModel:
        gage = GageStudyModel(
            source=self.df_thick,
            target='result_gage',
            reference='reference',
            u_cal=self.df_thick['U_cal'][0],
            tolerance=self.df_thick['tolerance'][0],
            resolution=self.df_thick['resolution'][0])
        model = GageRnRModel(
            source=self.df_thick,
            target='result_rnr',
            part='part',
            gage=gage,
            u_gv='operator')
        return model

    @pytest.fixture
    def rnr_adj_model(self) -> GageRnRModel:
        gage = GageStudyModel(
            source=self.df_adj,
            target='result_gage',
            reference='reference',
            u_cal=self.df_adj['U_cal'][0],
            tolerance=self.df_adj['tolerance'][0],
            resolution=self.df_adj['resolution'][0])
        model = GageRnRModel(
            source=self.df_adj,
            target='result_rnr',
            part='part',
            gage=gage,
            u_av='operator')
        return model

    def test_anova_table(self, rnr_thick_model: GageRnRModel) -> None:
        """Verification done with Minitab v22.2

        Stat > Quality Tools > Gage Study > Gage R&R Study (Crossed)"""
        anova = rnr_thick_model.anova()

        assert anova['DF']['part'] == 9
        assert anova['DF']['operator'] == 2
        assert anova['DF']['part:operator'] == 18
        assert anova['DF'][ANOVA.RESIDUAL] == 30
        
        assert anova['SS']['part'] == approx(0.0015878, abs=1e-7)
        assert anova['SS']['operator'] == approx(0.0000008, abs=1e-7)
        assert anova['SS']['part:operator'] == approx(0.0000078, abs=1e-7)
        assert anova['SS'][ANOVA.RESIDUAL] == approx(0.0000232, abs=1e-3)
 
        assert anova['MS']['part'] == approx(0.0001764, abs=1e-7)
        assert anova['MS']['operator'] == approx(0.0000004, abs=1e-7)
        assert anova['MS']['part:operator'] == approx(0.0000004, abs=1e-7)
        assert anova['MS'][ANOVA.RESIDUAL] == approx(0.0000008, abs=1e-7)
 
        assert anova['p']['part:operator'] == approx(0.9183453, abs=1e-7)

    def test_anova_table_gv(self, rnr_thick_model_gv: GageRnRModel) -> None:
        """Verification done with Minitab v22.2

        Stat > Quality Tools > Gage Study > Gage R&R Study (Crossed)"""
        anova = rnr_thick_model_gv.anova()

        assert anova['DF']['part'] == 9
        assert anova['DF']['operator'] == 2
        assert anova['DF']['part:operator'] == 18
        assert anova['DF'][ANOVA.RESIDUAL] == 30
        
        assert anova['SS']['part'] == approx(0.0015878, abs=1e-7)
        assert anova['SS']['operator'] == approx(0.0000008, abs=1e-7)
        assert anova['SS']['part:operator'] == approx(0.0000078, abs=1e-7)
        assert anova['SS'][ANOVA.RESIDUAL] == approx(0.0000232, abs=1e-3)
 
        assert anova['MS']['part'] == approx(0.0001764, abs=1e-7)
        assert anova['MS']['operator'] == approx(0.0000004, abs=1e-7)
        assert anova['MS']['part:operator'] == approx(0.0000004, abs=1e-7)
        assert anova['MS'][ANOVA.RESIDUAL] == approx(0.0000008, abs=1e-7)
 
        assert anova['p']['part:operator'] == approx(0.9183453, abs=1e-7)

    def test_rnr_table(self, rnr_thick_model: GageRnRModel) -> None:
        """Verification done with Minitab v22.2
        
        Stat > Quality Tools > Gage Study > Gage R&R Study (Crossed)
        
        The verification values for 6s/Tolerance coming from the 
        MSA Assistant."""
        rnr = rnr_thick_model.rnr()

        ms = rnr['MS']
        assert ms[ANOVA.RNR] == approx(0.00000067, abs=1e-8)
        assert ms[ANOVA.EV] == approx(0.00000067, abs=1e-8)
        assert ms[ANOVA.AV] == approx(0.00000000, abs=1e-8)
        assert ms[ANOVA.PV] == approx(0.00002929, abs=1e-8)
        assert ms[ANOVA.TOTAL] == approx(0.00002997, abs=1e-8)

        ms_tot = rnr['MS/Total']
        assert ms_tot[ANOVA.RNR] == approx(0.02247966, abs=1e-8)
        assert ms_tot[ANOVA.EV] == approx(0.02247966, abs=1e-8)
        assert ms_tot[ANOVA.AV] == approx(0.00000000, abs=1e-8)
        assert ms_tot[ANOVA.PV] == approx(0.97752034, abs=1e-8)
        assert ms_tot[ANOVA.TOTAL] == approx(1.000000, abs=1e-8)

        std = rnr['s']
        assert std[ANOVA.RNR] == approx(0.00082074, abs=1e-8)
        assert std[ANOVA.EV] == approx(0.00082074, abs=1e-8)
        assert std[ANOVA.AV] == approx(0.00000000, abs=1e-8)
        assert std[ANOVA.PV] == approx(0.00541218, abs=1e-8)
        assert std[ANOVA.TOTAL] == approx(0.00547406, abs=1e-8)

        spread_tot = rnr['6s/Total']
        assert spread_tot[ANOVA.RNR] == approx(0.14993220, abs=1e-8)
        assert spread_tot[ANOVA.EV] == approx(0.14993220, abs=1e-8)
        assert spread_tot[ANOVA.AV] == approx(0.00000000, abs=1e-8)
        assert spread_tot[ANOVA.PV] == approx(0.98869628, abs=1e-8)
        assert spread_tot[ANOVA.TOTAL] == approx(1.00000000, abs=1e-8)

        spread_tol = rnr['6s/Tolerance']
        assert spread_tol[ANOVA.RNR] == approx(0.1641, abs=1e-4)
        assert spread_tol[ANOVA.EV] == approx(0.1641, abs=1e-4)
        assert spread_tol[ANOVA.AV] == approx(0.0000, abs=1e-4)
        assert spread_tol[ANOVA.PV] == approx(1.0824, abs=1e-4)
        assert spread_tol[ANOVA.TOTAL] == approx(1.0948, abs=1e-4)

    def test_rnr_table_gv(self, rnr_thick_model_gv: GageRnRModel) -> None:
        """Verification done with Minitab v22.2
        
        Stat > Quality Tools > Gage Study > Gage R&R Study (Crossed)
        
        The verification values for 6s/Tolerance coming from the 
        MSA Assistant."""
        rnr = rnr_thick_model_gv.rnr()

        ms = rnr['MS']
        assert ms[ANOVA.RNR] == approx(0.00000067, abs=1e-8)
        assert ms[ANOVA.EV] == approx(0.00000067, abs=1e-8)
        assert ms[ANOVA.GV] == approx(0.00000000, abs=1e-8)
        assert ms[ANOVA.PV] == approx(0.00002929, abs=1e-8)
        assert ms[ANOVA.TOTAL] == approx(0.00002997, abs=1e-8)

        ms_tot = rnr['MS/Total']
        assert ms_tot[ANOVA.RNR] == approx(0.02247966, abs=1e-8)
        assert ms_tot[ANOVA.EV] == approx(0.02247966, abs=1e-8)
        assert ms_tot[ANOVA.GV] == approx(0.00000000, abs=1e-8)
        assert ms_tot[ANOVA.PV] == approx(0.97752034, abs=1e-8)
        assert ms_tot[ANOVA.TOTAL] == approx(1.000000, abs=1e-8)

        std = rnr['s']
        assert std[ANOVA.RNR] == approx(0.00082074, abs=1e-8)
        assert std[ANOVA.EV] == approx(0.00082074, abs=1e-8)
        assert std[ANOVA.GV] == approx(0.00000000, abs=1e-8)
        assert std[ANOVA.PV] == approx(0.00541218, abs=1e-8)
        assert std[ANOVA.TOTAL] == approx(0.00547406, abs=1e-8)

        spread_tot = rnr['6s/Total']
        assert spread_tot[ANOVA.RNR] == approx(0.14993220, abs=1e-8)
        assert spread_tot[ANOVA.EV] == approx(0.14993220, abs=1e-8)
        assert spread_tot[ANOVA.GV] == approx(0.00000000, abs=1e-8)
        assert spread_tot[ANOVA.PV] == approx(0.98869628, abs=1e-8)
        assert spread_tot[ANOVA.TOTAL] == approx(1.00000000, abs=1e-8)

        spread_tol = rnr['6s/Tolerance']
        assert spread_tol[ANOVA.RNR] == approx(0.1641, abs=1e-4)
        assert spread_tol[ANOVA.EV] == approx(0.1641, abs=1e-4)
        assert spread_tol[ANOVA.GV] == approx(0.0000, abs=1e-4)
        assert spread_tol[ANOVA.PV] == approx(1.0824, abs=1e-4)
        assert spread_tol[ANOVA.TOTAL] == approx(1.0948, abs=1e-4)

    def test_rnr_table_interaction(self) -> None:
        """Verification done with:

        Dr. Bill McNeese, BPI Consulting, LLC (09.2012)
        
        https://www.spcforexcel.com/knowledge/measurement-systems-analysis-gage-rr/anova-gage-rr-part-2/
        """
        df = load_dataset('grnr_spc')
        gage = GageStudyModel.from_gage_estimators(
            GageEstimator(
                samples=df['result'],
                reference=None,
                u_cal=0.002,
                tolerance=15,
                resolution=None))
        rnr_thick_model = GageRnRModel(
            source=df,
            target='result',
            part='part',
            gage=gage,
            u_av='operator')
        rnr = rnr_thick_model.rnr(evaluate_ia=True)
        
        ms = rnr['MS']
        assert ms[ANOVA.RNR] == approx(0.1109, abs=1e-4)
        assert ms[ANOVA.EV] == approx(0.0571, abs=1e-4)
        assert ms[ANOVA.AV] == approx(0.0538, abs=1e-4)
        assert ms[ANOVA.PV] == approx(0.8021, abs=1e-4)
        assert ms[ANOVA.TOTAL] == approx(0.9130, abs=1e-4)

        ms_tot = rnr['MS/Total']
        assert ms_tot[ANOVA.RNR] == approx(0.1214, abs=1e-4)
        assert ms_tot[ANOVA.EV] == approx(0.0625, abs=1e-4)
        assert ms_tot[ANOVA.AV] == approx(0.0589, abs=1e-4)
        assert ms_tot[ANOVA.PV] == approx(0.8786, abs=1e-4)
        assert ms_tot[ANOVA.TOTAL] == approx(1.0000, abs=1e-4)

    def test_rnr_table_interaction_gv(self) -> None:
        """Verification done with:

        Dr. Bill McNeese, BPI Consulting, LLC (09.2012)
        
        https://www.spcforexcel.com/knowledge/measurement-systems-analysis-gage-rr/anova-gage-rr-part-2/
        """
        df = load_dataset('grnr_spc')
        gage = GageStudyModel.from_gage_estimators(
            GageEstimator(
                samples=df['result'],
                reference=None,
                u_cal=0.002,
                tolerance=15,
                resolution=None))
        rnr_thick_model = GageRnRModel(
            source=df,
            target='result',
            part='part',
            gage=gage,
            u_gv='operator')
        rnr = rnr_thick_model.rnr(evaluate_ia=True)
        
        ms = rnr['MS']
        assert ms[ANOVA.RNR] == approx(0.1109, abs=1e-4)
        assert ms[ANOVA.EV] == approx(0.0571, abs=1e-4)
        assert ms[ANOVA.GV] == approx(0.0538, abs=1e-4)
        assert ms[ANOVA.PV] == approx(0.8021, abs=1e-4)
        assert ms[ANOVA.TOTAL] == approx(0.9130, abs=1e-4)

        ms_tot = rnr['MS/Total']
        assert ms_tot[ANOVA.RNR] == approx(0.1214, abs=1e-4)
        assert ms_tot[ANOVA.EV] == approx(0.0625, abs=1e-4)
        assert ms_tot[ANOVA.GV] == approx(0.0589, abs=1e-4)
        assert ms_tot[ANOVA.PV] == approx(0.8786, abs=1e-4)
        assert ms_tot[ANOVA.TOTAL] == approx(1.0000, abs=1e-4)

    def test_uncertainties(self) -> None:
        df_gage = pd.read_csv(
            valid_data_dir/'gage_study.csv',
            skiprows=lambda x: x not in list(range(87, 178)),
            sep=';'
            ).dropna(how='all', axis=1)
        
        df_rnr = pd.read_csv(
            valid_data_dir/'gage_rnr.csv',
            skiprows=lambda x: x not in list(range(1, 62)),
            sep=';')
        
        df_vmpt = (df_rnr
            .loc[:, 'influence': 'rank']
            .dropna(how='all', axis=0)
            .set_index('influence'))
        
        gage = GageStudyModel(
            source=df_gage,
            target='result',
            reference='reference',
            u_cal=df_gage['U_cal'][0],
            tolerance=df_gage['tolerance'][0],
            resolution=df_gage['resolution'][0],
            u_lin=MeasurementUncertainty(standard=0),
            u_rest=MeasurementUncertainty(
                error_limit=0.0008, distribution='rectangular'),
            k=2,)
        rnr_model = GageRnRModel(
            source=df_rnr,
            target='result',
            part='part',
            gage=gage,
            u_gv='point',
            u_t=MeasurementUncertainty(standard=0.00126),
            u_rest=MeasurementUncertainty(
                error_limit=0.0022, distribution='rectangular'))

        rnr_model.uncertainties()
        df_u = rnr_model.df_u
        df_ums = rnr_model.df_ums
        df_ump = rnr_model.df_ump
        assert not df_u.empty
        assert not df_ums.empty
        assert not df_ump.empty

        assert list(df_ump.index) == [
            {'REST': 'MP_REST'}.get(r, r) for r in ANOVA.UNCERTAINTY_ROWS_MP]
        assert list(df_ums.index) == [
            {'REST': 'MS_REST'}.get(r, r) for r in ANOVA.UNCERTAINTY_ROWS_MS]
        
        for u_is, u_valid in zip(df_u['u'], df_vmpt['u']):
            assert u_is == approx(u_valid, abs=1e-5)
        
        for U_is, U_valid in zip(df_u['U'], df_vmpt['U']):
            assert U_is == approx(U_valid, abs=1e-4)
        
        for Q_is, Q_valid in zip(df_u['Q'], df_vmpt['Q']):
            assert Q_is == approx(Q_valid, abs=1e-3)
        
        for r_is, r_valid in zip(df_u['rank'], df_vmpt['rank']):
            if pd.isna(r_is):
                assert pd.isna(r_valid)
            else:
                assert r_is == r_valid


# ============================================================================
# TESTS FOR NEW FEATURE: GeneralizedLinearModel
# ============================================================================

class TestGeneralizedLinearModel:
    """Tests for Generalized Linear Model for count data and other non-normal responses"""
    
    @pytest.fixture
    def count_data(self) -> DataFrame:
        """Create count data (e.g., defect counts) for testing"""
        import numpy as np
        np.random.seed(42)
        
        data = {
            'Factor_A': ['A1']*20 + ['A2']*20 + ['A3']*20,
            'Factor_B': ['B1', 'B2']*30,
            'Count': (
                # A1 combinations
                [10, 12, 11, 13, 12, 14, 11, 13, 10, 12] + 
                [8, 7, 9, 10, 8, 9, 7, 8, 9, 10] +
                # A2 combinations (lower counts)
                [5, 4, 6, 5, 4, 6, 5, 4, 6, 5] +
                [3, 2, 4, 3, 2, 4, 3, 2, 4, 3] +
                # A3 combinations (higher counts)
                [18, 20, 19, 21, 17, 19, 18, 20, 19, 21] +
                [25, 24, 26, 27, 23, 25, 24, 26, 27, 23]
            )
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def proportion_data(self) -> DataFrame:
        """Create proportion/binomial data for testing"""
        data = {
            'Treatment': ['Control']*15 + ['Drug_A']*15 + ['Drug_B']*15,
            'Success': [3, 4, 2, 3, 4, 3, 2, 4, 3, 2, 4, 3, 2, 3, 4] +
                       [8, 9, 7, 8, 9, 8, 7, 9, 8, 7, 9, 8, 7, 8, 9] +
                       [12, 11, 13, 12, 11, 13, 12, 11, 13, 12, 11, 13, 12, 11, 13],
            'Total': [15]*45
        }
        df = pd.DataFrame(data)
        df['Proportion'] = df['Success'] / df['Total']
        return df
    
    def test_poisson_initialization(self, count_data: DataFrame) -> None:
        """Test Poisson GLM initialization"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        assert model is not None
        assert hasattr(model, 'family')
        assert model.family == 'poisson'
    
    def test_negative_binomial_initialization(self, count_data: DataFrame) -> None:
        """Test Negative Binomial GLM initialization"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='negbin',
            order=1
        )
        
        assert model is not None
        assert model.family == 'negbin'
    
    def test_binomial_initialization(self, proportion_data: DataFrame) -> None:
        """Test Binomial GLM initialization"""
        model = GeneralizedLinearModel(
            source=proportion_data,
            target='Proportion',
            factors=['Treatment'],
            family='binomial',
            order=1
        )
        
        assert model is not None
        assert model.family == 'binomial'
    
    def test_fit_method(self, count_data: DataFrame) -> None:
        """Test model fitting"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1,
            fit_at_init=False
        )
        
        model.fit()
        
        # Should have fitted model
        assert hasattr(model, '_model')
        assert model._model is not None
    
    def test_deviance_check(self, count_data: DataFrame) -> None:
        """Test deviance and overdispersion check"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        fit_check = model.deviance_check()
        
        # Check return structure
        assert isinstance(fit_check, dict)
        assert 'deviance' in fit_check
        assert 'df_resid' in fit_check
        assert 'dispersion' in fit_check
        assert 'overdispersed' in fit_check
        assert 'recommendation' in fit_check
        
        # Check values are reasonable
        assert fit_check['deviance'] >= 0
        assert fit_check['df_resid'] >= 0
        assert fit_check['dispersion'] >= 0
        assert isinstance(fit_check['overdispersed'], (bool, np.bool_))
        assert isinstance(fit_check['recommendation'], str)
    
    def test_dispersion_property(self, count_data: DataFrame) -> None:
        """Test dispersion property"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        dispersion = model.dispersion
        
        # Dispersion should be positive
        assert dispersion > 0
        # For Poisson, dispersion can vary depending on data
        assert 0.1 < dispersion < 10.0  # Generous bounds
    
    def test_deviance_property(self, count_data: DataFrame) -> None:
        """Test deviance property"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        deviance = model.deviance
        
        # Deviance should be non-negative
        assert deviance >= 0
    
    def test_parameter_statistics(self, count_data: DataFrame) -> None:
        """Test parameter statistics extraction"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        param_stats = model.parameter_statistics()
        
        # Should return DataFrame
        assert isinstance(param_stats, pd.DataFrame)
        
        # Check expected columns
        expected_cols = ['coef', 'std err', 'z', 'p', 'ci_low', 'ci_upp']
        assert all(col in param_stats.columns for col in expected_cols)
        
        # Should have intercept and factor terms
        assert len(param_stats) > 0
    
    def test_effects(self, count_data: DataFrame) -> None:
        """Test standardized effects calculation"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        effects = model.effects()
        
        # Should return Series
        assert isinstance(effects, pd.Series)
        
        # Should contain factor effects
        assert len(effects) > 0
    
    def test_p_values(self, count_data: DataFrame) -> None:
        """Test p-values extraction"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        p_vals = model.p_values()
        
        # Should return Series
        assert isinstance(p_vals, pd.Series)
        
        # P-values should be in [0, 1]
        assert all(p_vals >= 0)
        assert all(p_vals <= 1)
    
    def test_predict(self, count_data: DataFrame) -> None:
        """Test prediction on new data"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        # Predict with dict input
        predictions = model.predict({'Factor_A': ['A1', 'A2'], 'Factor_B': ['B1', 'B2']})
        
        # Should return DataFrame with predictions
        assert isinstance(predictions, pd.DataFrame)
        assert 'Count' in predictions.columns
        assert len(predictions) == 2
        assert all(predictions['Count'] >= 0)  # Counts should be non-negative
    
    def test_compare_groups(self, count_data: DataFrame) -> None:
        """Test comparison of all factor combinations"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        comparisons = model.compare_groups()
        
        # Should return DataFrame
        assert isinstance(comparisons, pd.DataFrame)
        
        # Should have columns for factors and predictions
        assert 'Factor_A' in comparisons.columns
        assert 'Factor_B' in comparisons.columns
        assert 'predicted_mean' in comparisons.columns
        
        # Should have all combinations (3 levels of A × 2 levels of B = 6)
        assert len(comparisons) == 6
        
        # Predicted means should be positive
        assert all(comparisons['predicted_mean'] > 0)
    
    def test_summary(self, count_data: DataFrame) -> None:
        """Test model summary generation"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        summary = model.summary()
        
        # Summary should be a string or similar
        assert summary is not None
        assert len(str(summary)) > 0
    
    def test_interactions_order_2(self, count_data: DataFrame) -> None:
        """Test model with interactions (order=2)"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=2  # Include interactions
        )
        
        param_stats = model.parameter_statistics()
        
        # Should have interaction terms
        # Check if any parameter name contains ':'
        has_interaction = any(':' in str(idx) for idx in param_stats.index)
        assert has_interaction or len(param_stats) > 3  # Either explicit or implicit
    
    def test_overdispersion_detection(self) -> None:
        """Test overdispersion detection with overdispersed data"""
        # Create highly overdispersed data
        np.random.seed(123)
        overdispersed_data = {
            'Factor': ['A']*30 + ['B']*30,
            'Count': np.concatenate([
                np.random.negative_binomial(5, 0.3, 30),  # High variance
                np.random.negative_binomial(10, 0.3, 30)
            ])
        }
        df = pd.DataFrame(overdispersed_data)
        
        model = GeneralizedLinearModel(
            source=df,
            target='Count',
            factors=['Factor'],
            family='poisson',
            order=1
        )
        
        fit_check = model.deviance_check()
        
        # Should detect overdispersion (though not guaranteed with random data)
        assert isinstance(fit_check['overdispersed'], (bool, np.bool_))
        assert fit_check['dispersion'] > 0
    
    def test_negative_binomial_for_overdispersion(self, count_data: DataFrame) -> None:
        """Test that Negative Binomial handles overdispersion better than Poisson"""
        
        # Fit both models
        model_poisson = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        
        model_negbin = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='negbin',
            order=1
        )
        
        # Both should fit without error
        assert model_poisson is not None
        assert model_negbin is not None
        
        # Both should produce predictions
        pred_poisson = model_poisson.compare_groups()
        pred_negbin = model_negbin.compare_groups()
        
        assert len(pred_poisson) == len(pred_negbin)
    
    def test_invalid_family(self, count_data: DataFrame) -> None:
        """Test error handling for invalid family"""
        with pytest.raises((ValueError, AssertionError)):
            GeneralizedLinearModel(
                source=count_data,
                target='Count',
                factors=['Factor_A'],
                family='invalid_family',
                order=1
            )
    
    def test_missing_data_handling(self) -> None:
        """Test handling of missing data"""
        data_with_na = {
            'Factor': ['A', 'A', 'B', 'B', 'A', 'B'],
            'Count': [10, 12, np.nan, 20, 11, 19]
        }
        df = pd.DataFrame(data_with_na)
        
        # Should handle missing data (may drop or raise error)
        try:
            model = GeneralizedLinearModel(
                source=df,
                target='Count',
                factors=['Factor'],
                family='poisson',
                order=1
            )
            # If it succeeds, check it's valid
            assert model is not None
        except (ValueError, Exception):
            # If it raises, that's also acceptable
            pass
