import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from typing import Any
from typing import Set
from typing import Self
from typing import List
from typing import Dict
from typing import Literal
from typing import LiteralString
from typing import Generator
from patsy.desc import ModelDesc
from numpy.typing import NDArray
from pandas.core.frame import DataFrame
from patsy.design_info import DesignInfo
from pandas.core.series import Series
from scipy.optimize._optimize import OptimizeResult
from statsmodels.iolib.table import Cell
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import forg
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.summary import summary_params_frame
from statsmodels.iolib.tableformatting import fmt_base
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .utils import uniques
from .utils import hierarchical
from .utils import get_term_name
from .utils import is_main_feature

from ..constants import ANOVA


class LinearModel:
    """This class is used to create and simplify linear models so that 
    only significant features describe the model.
    
    Balanced models (DOEs or EVOPs) including continuous can be 
    analyzed. With this class, you can create an encoded design matrix
    with all factor levels, including their interactions. All
    non-significant factors can then be automatically eliminated.
    Furthermore, this class allows the examination of main effects, 
    the sum of squares (explained variation), and the Anova table in 
    more detail.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas DataFrame as tabular data in a long format used for the
        model.
    target : str
        Column name of the endogenous variable.
    categorical : List[str]
        Column names of the categorical exogenous variables.
    continuous : List[str], optional
        Column names for continuous exogenous variables.
    alpha : float, optional
        Threshold as alpha risk. All features, including continuous and 
        intercept, that have a p-value smaller than alpha are removed 
        during the automatic elimination of the factors. Default is 0.05.
    """
    __slots__ = (
        'data', 'target', 'categorical', 'continuous', '_model', '_alpha',
        'input_map', 'input_rmap', 'output_map', 'exclude', 'level_map',
        '_initial_terms_', '_p_values', '_anova', '_effects', '_vif')
    data: DataFrame
    target: str
    categorical: List[str]
    continuous: List[str]
    _alpha: float
    _model: RegressionResultsWrapper | None
    input_map: Dict[str, str]
    input_rmap: Dict[str, str]
    output_map: Dict[str, str]
    exclude: Set[str]
    level_map: Dict[Any, Any]
    _initial_terms_: List[LiteralString]
    _p_values: 'Series[float]'
    _anova: DataFrame
    _effects: Series
    _vif: 'Series[float]'

    def __init__(
            self,
            source: DataFrame,
            target: str,
            categorical: List[str],
            continuous: List[str] = [],
            alpha: float = 0.05,
            order: int = 1) -> None:
        assert order >= 0 and isinstance(order, int), (
            'Interaction order must be a positive integer')
        self.target = target
        self.categorical = categorical
        self.continuous = continuous
        self.output_map = {target: 'y'}
        _categorical = tuple(f'x{i}' for i in range(len(categorical)))
        _continuous = tuple(f'e{i}' for i in range(len(continuous)))
        self.input_map = (
            {f: _f for f, _f in zip(categorical, _categorical)}
            | {c: _c for c, _c in zip(continuous, _continuous)})
        self.input_rmap = {v: k for k, v in self.input_map.items()}
        self.alpha = alpha
        self.exclude = set()
        self._model = None
        self.data = (source
            .copy()
            .rename(columns=self.input_map|self.output_map))
        model_desc = ModelDesc.from_formula(
            f'{self.output_map[self.target]}~'
            + ('*'.join(_categorical))
            + ('+'.join(['', *_continuous]) if _continuous else ''))
        terms = model_desc.describe().split(' ~ ')[1].split(' + ')
        self._initial_terms_ = [
            t for t in terms if len(t.split(ANOVA.SEP)) <= order]
        self._reset_tables_()
        
    @property
    def model(self) -> RegressionResultsWrapper:
        """Get regression results of fitted model. Raises AssertionError
        if no model is fitted yet (read-only)."""
        assert self._model is not None, (
            'Model not fitted yet, call `fit` method first.')
        return self._model

    @property
    def initial_formula(self) -> str:
        initial_formula = (
            f'{self.output_map[self.target]} ~ '
            + ' + '.join(self._initial_terms_))
        return initial_formula
    
    @property
    def uncertainty(self) -> float:
        """Get uncertainty of the model as square root of MS_Residual
        (read-only)."""
        if self._anova.empty:
            return np.nan
        else:
            return self._anova['MS']['Residual']**0.5

    @property
    def alpha(self) -> float:
        """Alpha risk as significance threshold for p-value of exegenous
        factors."""
        return self._alpha
    @alpha.setter
    def alpha(self, alpha: float) -> None:
        assert 0 < alpha < 1, 'Alpha risk must be between 0 and 1'
        self._alpha = alpha
    
    @property
    def design_info(self) -> DesignInfo:
        """Get the DesignInfo instance of current fitted model
        (read-only)."""
        return self.model.model.data.design_info
    
    @property
    def _term_names_(self) -> List[str]:
        """Get the internal names of all terms variables for the current 
        fitted model (read-only)."""
        return self.model.model.data.design_info.term_names
    
    @property
    def term_names(self) -> List[str]:
        """Get the names of all terms variables for the current fitted 
        model (read-only)."""
        return list(map(self._convert_term_name_, self._term_names_))

    @property
    def _term_map_(self) -> Dict[str, str]:
        """Get the names of all internal used and original terms 
        variables for the current fitted model as dict (read-only)"""
        return {n: self._convert_term_name_(n) for n in self._term_names_}
    
    @property
    def main_features(self) -> List[str]:
        """Get all main parameters of current model excluding intercept
        (read-only)."""
        return [n for n in self._term_names_ if is_main_feature(n)]
    
    @property
    def formula(self) -> str:
        """Get the formula used for the linear model, excluding any
        factors specified in the `exclude` attribute. The formula is
        constructed based on the excluded factors. If the intercept is
        excluded, the formula will include '-1' as the first term.
        Otherwise, the formula will include the original terms
        (excluding the excluded factors) separated by '+' (read-only)."""
        if self._model is None:
            return self.initial_formula
    
        ignore = list(self.exclude) + [ANOVA.INTERCEPT]
        terms = [t for t in self._initial_terms_ if t not in ignore]
        if ANOVA.INTERCEPT in self.exclude:
            terms = ['-1'] + terms
        return f'{self.output_map[self.target]} ~ {" + ".join(terms)}'
    
    def _reset_tables_(self) -> None:
        """Reset the anova table, the p_values and the effects."""
        self._anova = pd.DataFrame()
        self._effects = pd.Series()
        self._p_values = pd.Series()
        self._vif = pd.Series()
    
    def _convert_single_term_name_(self, term_name: str) -> str:
        """Convert the single term name using the original names stored
        in `input_rmap`.

        Parameters
        ----------
        term_name : str
            The term name (no interaction).

        Returns
        -------
        str
            The converted term name.
        """
        split = term_name.split('[T.')
        split[0] = self.input_rmap.get(split[0], split[0])
        return '[T.'.join(split)

    def _convert_term_name_(self, term_name: str) -> str:
        """Convert the term name using the categorical or continuous 
        names provided when initializing.

        Parameters
        ----------
        term_name : str
            The term name of the design info.

        Returns
        -------
        str
            The converted term name.
        """
        if term_name == ANOVA.INTERCEPT:
            return term_name

        converted_name = ANOVA.SEP.join(map(
            self._convert_single_term_name_,
            term_name.split(ANOVA.SEP)))
        return converted_name
    
    def is_hierarchical(self) -> bool:
        """Check if current fitted model is hierarchical."""
        hierarchical_terms = hierarchical(self._term_names_)
        return all([term in self._term_names_ for term in hierarchical_terms])
    
    def effects(self) -> Series:
        """Calculates the impact of each term on the target. The
        effects are described as twice the parameter coefficients and 
        occur as an absolute number."""
        if self._effects.empty:
            params: Series = 2 * self.model.params
            names_map = {n: get_term_name(n) for n in params.index}
            self._effects = (params
                .abs()
                .rename(index=names_map)
                .groupby(level=0, axis=0)
                .sum()
                [uniques(names_map.values())])
            self._effects.name = ANOVA.EFFECTS
            self._effects.index.name = ANOVA.FEATURES
        effects = self._effects.copy().rename(index=self._term_map_)
        return effects
    
    def p_values(self) -> 'Series[float]':
        """Get P-value for significance of adding model terms using 
        anova typ III table for current model."""
        if self._p_values.empty:
            anova = self.anova(typ='III')
            if anova.empty:
                self._p_values = pd.Series(
                    {t: np.nan for t in self.design_info.term_names})
            else:
                self._p_values = self._anova['p'].iloc[:-1]
        return self._p_values.copy().rename(index=self._term_map_)

    def least_term(self) -> str:
        """Get the term name with the least effect or the least p-value
        coming from a ANOVA typ III table of current fitted model.

        Returns
        -------
        str
            The term name with the least effect or the least p-value.

        Notes
        -----
        This method checks if any p-values are missing (NaN). If there
        are missing p-values, it returns the term name that has the 
        smallest effect on the target variable. Otherwise, it returns
        the term name with the least p-value for the F-stats coming from
        current ANOVA table.
        """
        if any(self.p_values().isna()):
            self.effects()
            smallest = np.where(self._effects == self._effects.min())[0][-1]
            least = str(self._effects.index[smallest])
        else:
            least = str(self._p_values.index[self._p_values.argmax()])
        return least
    
    def fit(self, **kwds) -> Self:
        """Create and fit a ordinary least squares model using current 
        formula. Then  Finally calculate 
        the impact of each term on the target.
        
        Parameters
        ----------
        **kwds
            Additional keyword arguments for `ols` function of 
            `statsmodels.formula.api`.
        """
        self._reset_tables_()
        formula = kwds.pop('formula', self.formula)
        self._model = smf.ols(formula, self.data, **kwds).fit()
        return self
    
    def gof_metrics(self, index: int | str = 0) -> DataFrame:
        """Get different goodness-of-fit metrics (read-only).

        Parameters
        ----------
        index : int | str
            Value is set as index. When using the method 
            recursive_feature_elimination, the current step is passed as
            index
        
        Returns
        -------
        DataFrame
            The goodness-of-fit metrics table as DataFrame containing
            the following columns:
            - 'formula' = current formula
            - 's' = Uncertainty of the model as square root of MS_Residual
            - 'aic' = Akaike's information criteria
            - 'r2' = R-squared of the model
            - 'r2_adj' = adjusted R-squared
            - 'least_term' = the least significant term
            - 'p_least' = The p-value of least significant term, coming
            from ANOVA table Type-III.
            - 'hierarchical' = True if model is hierarchical
        """
        self.anova(typ='III', vif=False)
        data = {
            'formula': self.formula,
            's': self.uncertainty,
            'aic': self.model.aic,
            'r2': self.model.rsquared,
            'r2_adj': self.model.rsquared_adj,
            'least_term': self._convert_term_name_(self.least_term()),
            'p_least': self.p_values().max(),
            'hierarchical': self.is_hierarchical()}
        return pd.DataFrame(data, index=[index])
    
    def summary(
            self, 
            anova_typ: Literal['', 'I', 'II', 'III', None] = None,
            vif: bool = True, **kwds
            ) -> Summary:
        """Generate a summary of the fitted model.

        Parameters
        ----------
        vif : bool, optional
            If True, variance inflation factors (VIF) are added to the 
            anova table. Will only be considered if anova_typ is not 
            None, by default True
        anova_typ: Literal['', 'I', 'II', 'III', None] , optional
            If not None, add an ANOVA table of provided type to the 
            summary, by default None.
        **kwds
            Additional keyword arguments to be passed to the `summary` 
            method of
            `statsmodels.regression.linear_model.RegressionResults` 
            class.

        Returns
        -------
        Summary
            A summary object containing information about the fitted 
            model.
        """
        _kwds = dict(
            yname=self.target,
            xname=list(map(self._convert_term_name_, self.model.params.index))
            ) | kwds
        summary: Summary = self.model.summary(**_kwds)
        summary.tables = [summary.tables[i] for i in [0, 2, 1]]
   
        if isinstance(anova_typ, str):
            anova = self.anova(typ=anova_typ, vif=vif).map(forg)
            anova['DF'] = anova['DF'].astype(float).astype(int).astype(str)
            table = SimpleTable(
                data=anova.values,
                headers=anova.columns.to_list(),
                stubs=anova.index.to_list(),
                txt_fmt=fmt_base)
            summary.tables[0].title = (
                f'{summary.tables[0].title} (ANOVA {anova.columns.name})')
            summary.tables.append(table)
        return summary

    def recursive_feature_elimination(
            self, rsquared_max: float = 0.99, ensure_hierarchy: bool = True,
            **kwds) -> Generator[DataFrame, Any, None]:
        """Perform a linear regression starting with complete model.
        Then recursive features are eliminated according to the highest
        p-value (two-tailed p values for the t-stats of the params).
        Features are eliminated until only significant features
        (p-value < given threshold) remain in the model.

        Parameters
        ----------
        rsquared_max : float in (0, 1), optional
            If given, the model must have a lower R^2 value than the 
            given threshold, by default 0.99
        ensure_hierarchy : bool, optional
            Adds features at the end to ensure model is hierarchical, 
            by default True
        **kwds
            Additional keyword arguments for `ols` function of 
            `statsmodels.formula.api`.

        Notes
        -----
        The attribute `exclude` is reset at the beginning and then
        refilled.
        """
        self._model = None
        self.exclude = set()
        self.fit(**kwds)
        max_steps = len(self._term_names_)
        step = -1
        for step in range(max_steps):
            if self.has_insignificant_term(rsquared_max):
                self.exclude.add(self.least_term())
                self.fit(**kwds)
                yield self.gof_metrics(step)
            else:
                break
        
        if step < 1:
            yield self.gof_metrics(step)
        
        if ensure_hierarchy and not self.is_hierarchical():
            step = step + 1
            h_features = hierarchical(self._term_names_)
            self.exclude = {e for e in self.exclude if e not in h_features}
            self.fit(**kwds)
            yield self.gof_metrics(step)
    
    def anova(
            self, typ: Literal['', 'I', 'II', 'III'] = '', vif: bool = False
            ) -> DataFrame:
        """Perform an analysis of variance (ANOVA) on the fitted model.

        Parameters
        ----------
        typ : Literal['', 'I', 'II', 'III'], optional
            The type of ANOVA to perform. Default is 'III', see notes
            for more informations about the types.
            - '' : If no or an invalid type is specified, Type-II is 
            used if the model has no significant interactions. 
            Otherwise, Type-III is used for hierarchical models and 
            Type-I is used for non-hierarchical models.
            - 'I' : Type I sum of squares ANOVA.
            - 'II' : Type II sum of squares ANOVA.
            - 'III' : Type III sum of squares ANOVA.
        vif : bool, optional
            If True, variance inflation factors (VIF) are added to the 
            anova table, by default False

        Returns
        -------
        DataFrame
            The ANOVA table as DataFrame containing the following
            columns:
            - DF : Degrees of freedom for model terms.
            - SS : Sum of squares for model terms.
            - F : F statistic value for significance of adding model
            terms.
            - p : P-value for significance of adding model terms.
            - n2 : Eta-square as effect size (proportion of explained
            variance).
            - np2 : Partial eta-square as partial effect size.

        Notes
        -----        
        The ANOVA table provides information about the significance of 
        each factor and interaction in the model. The type of ANOVA 
        determines how the sum of squares is partitioned among the 
        factors.

        The SAS and also Minitab software uses Type III by default. This
        type is also the only one who gives us a SS and p-value for the 
        Intercept. So Type-III is also used internaly for evaluating the
        least significant term. A discussion on which one to use can be 
        found here:
        https://stats.stackexchange.com/a/93031

        A nice conclusion about the differences between the types:
        - Typ-I: We choose the most "important" independent variable and 
        it will receive the maximum amount of variation possible.
        - Typ-II: We ignore the shared variation: no interaction is
        assumed. If this is true, the Type II Sums of Squares are
        statistically more powerful. However if in reality there is an
        interaction effect, the model will be wrong and there will be a
        problem in the conclusions of the analysis.
        - Typ-III: If there is an interaction effect and we are looking 
        for an “equal” split between the independent variables, 
        Type-III should be used.

        source:
        https://towardsdatascience.com/anovas-three-types-of-estimating-sums-of-squares-don-t-make-the-wrong-choice-91107c77a27a

        Examples
        --------
        >>> import daspi
        >>> from daspi.anova import LinearModel
        >>> df = daspi.load_dataset('anova3')
        >>> lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug']).fit()
        >>> print(lm.anova(typ='III').round(3))
        Typ-III    DF       SS       MS        F      p     n2
        source                                                
        Intercept   1  390.868  390.868  453.467  0.000  0.864
        Sex         1    2.075    2.075    2.407  0.127  0.005
        Risk        1   11.332   11.332   13.147  0.001  0.025
        Drug        2    0.816    0.408    0.473  0.626  0.002
        Residual   55   47.407    0.862      NaN    NaN  0.105
        """
        column_name = self._anova.columns.name
        if column_name and column_name.split('-')[1] != typ:
            self._reset_tables_()
        
        if self._anova.empty:
            if all(self.model.pvalues.isna()):
                self._anova = pd.DataFrame()
                warnings.warn(
                    'ANOVA table could not be calculated because the model is '
                    'underdetermined.')
                return self._anova
            
            if typ not in ('I', 'II', 'III'):
                if not self.has_significant_interactions():
                    typ = 'II'
                elif self.is_hierarchical():
                    typ = 'III'
                else:
                    typ = 'I'
            anova = sm.stats.anova_lm(self.model, typ=typ)
            anova = anova.rename(
                columns={'df': 'DF', 'sum_sq': 'SS', 'PR(>F)': 'p'})
            factors = [i for i in anova.index if i != 'Residual']
            ss_resid = anova.loc['Residual', 'SS']
            ss_factors = anova.loc[factors,'SS']

            anova['DF'] = anova['DF'].astype(int)
            anova['MS'] = anova['SS']/anova['DF']
            anova['n2'] = anova['SS'] / anova['SS'].sum()
            anova.index.name = ANOVA.SOURCE
            anova.columns.name = f'Typ-{typ}'
            self._anova = anova[ANOVA.TABLE_COLNAMES]
        
        if vif:
            _vif = self.variance_inflation_factor()
            indices = [i for i in self._vif.index if i in self._anova.index]
            self._anova.loc[indices, 'VIF'] = self._vif
        anova = self._anova.copy()
        anova.index = anova.index.map(self._convert_term_name_)
        return anova
    
    def parameter_statistics(
            self, alpha: float = 0.05, use_t: bool = True) -> DataFrame:
        """Calculate the parameter statistics for the fitted model.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence intervals, by 
            default 0.05.
        use_t : bool, optional
            If True, use t-distribution for hypothesis testing and 
            confidence intervals. If False, use normal distribution, 
            by default True.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the parameter statistics for 
            the fitted model. The DataFrame includes columns for the 
            parameter estimates, standard errors, t-values 
            (or z-values), and p-values.
        """
        params_table = summary_params_frame(
            self.model,
            yname=self.target,
            xname=list(map(self._convert_term_name_, self.model.params.index)),
            alpha=alpha,
            use_t=use_t)
        columns_map = {
            'P>|t|': 'p',
            'Conf. Int. Low': 'ci_low',
            'Conf. Int. Upp.': 'ci_upp'}
        params_table = params_table.rename(columns=columns_map)
        return params_table
    
    def variance_inflation_factor(self) -> Series:
        """Calculate the variance inflation factor (VIF) for each 
        predictor variable in the fitted model.

        Returns
        -------
        Series
            A pandas Series containing the VIF values for each predictor 
            variable.
        """
        param_names = self.model.model.data.param_names
        names_map = {n: get_term_name(n) for n in param_names}
        xs: NDArray = self.model.model.data.exog.copy()
        vif: Dict[str, float] = {}
        
        if self._vif.empty:
            for pos, name in enumerate(param_names):
                x = xs[:, pos]
                _xs = np.delete(xs, pos, axis=1)
                r2 = sm.OLS(x, _xs).fit().rsquared
                vif[name] = (1 / (1 - r2))
            self._vif = pd.Series(vif).rename(index=names_map)
            self._vif = self._vif[~self._vif.index.duplicated()]
        return self._vif.copy().rename(index=names_map)

    def highest_features(self) -> List[str]:
        """Get all main and interaction features that do not appear in a 
        higher interaction. Covariates are not taken into account here."""
        _features = [f for f in self.term_names if not f.startswith('e')]
        features_splitted = sorted(
            [f.split(ANOVA.SEP) for f in _features], 
            key=len, reverse=True)
        
        features = []
        highest_level = len(features_splitted[0])
        for f_split in features_splitted:
            level = len(f_split)
            if level == highest_level:
                features.append(f_split)
            else:
                intersect = [i for f in features for i in set(f_split) & set(f)]
                if len(intersect) < level:
                    features.append(f_split)
        return [ANOVA.SEP.join(f) for f in features]
    
    def has_insignificant_term(self, rsquared_max: float = 0.99) -> bool:
        """Check if the fitted model has any insignificant terms.

        Parameters
        ----------
        rsquared_max : float in (0, 1), optional
            The maximum R^2 value that the model can have to be 
            considered significant. If not provided, by default 0.99.

        Returns
        -------
        bool
            Returns True if the model has any insignificant terms, and 
            False otherwise.
        """
        if len(self._term_names_) == 1:
            return False
        
        if all(self.p_values().isna()):
            return True
        
        has_insignificant = (
            self._p_values.max() > self.alpha
            or self.model.rsquared > rsquared_max)
        return has_insignificant

    def has_significant_interactions(self) -> bool:
        """True if fitted model has significant interactions."""
        for feature in self.highest_features():
            if ANOVA.SEP not in feature:
                continue
            if self.p_values()[feature] < self.alpha:
                return True
        return False
    
    def predict(
            self, xs: List[float], intercept: Literal[0, 1] = 1,
            negate: bool = False) -> float:
        """Predict y with given xs. Ensure that all non interactions are 
        given in xs
        
        Parameters
        ----------
        xs : array_like
            The values for which you want to predict. Make sure the 
            order matches the `main_features` property.
        intercept : Literal[0, 1], optional
            Factor level for the intercept, either 0 or 1, by default 1
        negate : bool, optional
            If True, the predicted value is negated (used for 
            optimization), by default False
            
        Returns
        -------
        y : float
            Predicted value
        """
        assert len(xs) == len(self.main_features), (
            f'Please provide a value for each main feature')
        
        X = np.zeros(len(self._term_names_))
        for i, feature in enumerate(self._term_names_):
            if ANOVA.SEP not in feature:
                X[i] = xs[i]
        X[-1] = intercept
        y = float(self.model.predict(pd.DataFrame([X], columns=features))) # type: ignore
        return -y if negate else y
    
    def residual_data(self) -> DataFrame:
        """
        Get the residual data from the fitted model.

        Returns
        -------
        pd.DataFrame
            The residual data containing the residuals, observation index, and predicted values.

        Examples
        --------
        >>> import daspi
        >>> df = daspi.load_dataset('partial_factorial')
        >>> target = 'Yield'
        >>> features = [c for c in df.columns if c != target]
        >>> lm = LinearModel(df, target, features).fit()
        >>> print(lm.residual_data())
            Observation      Residues  Prediction
        0             0  9.250000e+00       46.75
        1             1  2.000000e+00       51.00
        2             2 -1.050000e+01       73.50
        3             3 -2.500000e-01       65.25
        4             4 -1.421085e-14       53.00
        5             5  1.025000e+01       44.75
        6             6 -2.500000e-01       67.25
        7             7 -1.050000e+01       71.50
        8             8  3.750000e+00       65.25
        9             9 -1.200000e+01       57.00
        10           10 -1.500000e+00       79.50
        11           11  9.250000e+00       83.75
        12           12 -1.000000e+01       59.00
        13           13 -3.250000e+00       63.25
        14           14  9.250000e+00       85.75
        15           15  4.500000e+00       77.50
        """
        data = self.model.resid
        data.name = 'Residues'
        data.index.name = 'Observation'
        data = data.to_frame().reset_index()
        data['Prediction'] = self.model.predict()
        return data