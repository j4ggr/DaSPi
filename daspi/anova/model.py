"""The `LinearModel` class in `daspi/anova/model.py` is designed to 
create and simplify linear models, where only significant features are 
used to describe the model. It is particularly useful for analyzing 
balanced models (DOEs or EVOPs) that include both categorical and 
continuous variables.

The class takes three main inputs:

- source: A pandas DataFrame containing the tabular data in a long 
format, which will be used for the model.
- target: The name of the column in the DataFrame that represents the 
endogenous (dependent) variable.
- categorical: A list of column names in the DataFrame that represent 
the categorical exogenous (independent) variables.
- Additionally, it can take an optional input continuous, which is a 
list of column names representing continuous exogenous variables.

The purpose of this class is to create a linear model that includes all 
factor levels and their interactions, and then automatically eliminate 
any non-significant factors or interactions. This allows for a more 
concise and accurate representation of the model, where only the 
relevant features are included.

To achieve this, the class follows these steps:

1. It encodes the design matrix with all factor levels and their 
interactions.
2. It fits a linear model using the encoded design matrix and the 
provided data.
3. It calculates the p-values for each factor and interaction,
indicating their significance in the model.
4. It provides methods to recursively eliminate non-significant factors 
or interactions based on their p-values, until only significant features 
remain.

The class also provides methods to analyze the model in more detail, 
such as:

- Calculating the sum of squares (explained variation) for each factor 
and interaction.
- Generating an ANOVA (Analysis of Variance) table to assess the 
significance of each factor and interaction.
- Calculating the effects (impact) of each factor and interaction on the 
target variable.
- Checking if the model is hierarchical (i.e., if all lower-order 
interactions are included when higher-order interactions are present).
- The main output of this class is a simplified linear model that 
includes only the significant features. Additionally, it provides 
various statistics and metrics related to the model, such as the ANOVA 
table, p-values, effects, and goodness-of-fit measures 
(e.g., R-squared, adjusted R-squared, AIC).

The class achieves its purpose through a combination of linear 
regression techniques, statistical hypothesis testing, and recursive 
feature elimination algorithms. It leverages the statsmodels library for 
fitting the linear models and performing ANOVA calculations.

Overall, the LinearModel class is a powerful tool for analyzing and 
simplifying linear models, particularly in the context of designed 
experiments or engineering applications where categorical and continuous 
variables are involved.
"""
import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats
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
from numpy.linalg import LinAlgError
from pandas.core.frame import DataFrame
from patsy.design_info import DesignInfo
from pandas.core.series import Series
from scipy.optimize._optimize import OptimizeResult
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import forg
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.summary import summary_params_frame
from statsmodels.iolib.tableformatting import fmt_base
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .convert import get_term_name
from .convert import frames_to_html

from .tables import anova_table
from .tables import terms_effect
from .tables import terms_probability
from .tables import variance_inflation_factor

from ..constants import ANOVA

from ..strings import STR


def is_main_feature(feature: str) -> bool:
    """Check if given feature is a main parameter (intercept is 
    excluded)."""
    return feature != ANOVA.INTERCEPT and ANOVA.SEP not in feature


def hierarchical(features: List[str]) -> List[str]:
    """Get all features such that all lower interactions and main 
    effects are present with the same features that appear in the higher 
    interactions

    Parameters
    ----------
    features : list of str
        Columns of exogene variables
    
    Returns
    -------
    h_features : list of str
        Sorted features for hierarchical model"""

    h_features = set(features)
    for feature in features:
        split = feature.split(ANOVA.SEP)
        n_splits = len(split)
        for s in split:
            h_features.add(s)
        if n_splits <= ANOVA.SMALLEST_INTERACTION:
            continue

        for i in range(ANOVA.SMALLEST_INTERACTION, n_splits):
            for combo in map(ANOVA.SEP.join, itertools.combinations(split, i)):
                h_features.add(combo)

    return sorted(sorted(list(h_features)), key=lambda x: x.count(ANOVA.SEP))


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
    skip_intercept_as_least : bool, optional
        If True, the intercept is not removed as a least significant 
        term when using recursive feature elimination. Also if True, the
        intercept does not appear when calling the `least_term` method,
        by default True.
    encode_categorical: bool, optional
        If True, all of the categorical variables are encoded using
        one-hot encoding. Otherwise they are interpreted as continuous
        variables when possible, by default True.
    
    Notes
    -----
    Always be careful when removing the intercept. If the intercept is 
    missing, Patsy will automatically add all one-hot encoded levels for
    a categorical variable to compensate for this missing term. This 
    will result in extremely high VIFs.
    """
    __slots__ = (
        'data', 'target', 'categorical', 'continuous', '_model', '_alpha',
        'skip_intercept_as_least', 'generalized_vif', 'input_map', 'input_rmap',
        'output_map', 'exclude', 'level_map', '_initial_terms_', '_p_values',
        '_anova', '_effects', '_vif')
    data: DataFrame
    """The Pandas DataFrame containing the data for the linear model."""
    target: str
    """The name of the target variable for the linear model."""
    categorical: List[str]
    """The list of categorical feature names used in the linear model."""
    continuous: List[str]
    """The list of continuous exogenous variables used in the linear 
    model."""
    _alpha: float
    """The alpha risk threshold used for automatic elimination of 
    features during model simplification. All features, including 
    continuous and intercept, that have a p-value smaller than this 
    alpha are removed from the model."""
    skip_intercept_as_least: bool
    """If True, the intercept is not treated as a least significant term"""
    generalized_vif: bool
    """If True, the generalized VIF is calculated when possible.
    Otherwise, only the straightforward VIF (via R2) is calculated."""
    _model: RegressionResultsWrapper | None
    """The regression results of the fitted model. This property raises 
    an AssertionError if no model has been fitted yet."""
    input_map: Dict[str, str]
    """A dictionary that maps the original feature names to the encoded 
    feature names used in the model."""
    input_rmap: Dict[str, str]
    """A dictionary that maps the encoded feature names back to the 
    original feature names used in the model."""
    output_map: Dict[str, str]
    """A dictionary that maps the original feature names to the encoded 
    feature names used in the model."""
    exclude: Set[str]
    """A set of feature names that should be excluded from the model."""
    level_map: Dict[Any, Any]
    """A dictionary that maps the original feature names to their 
    encoded versions used in the model."""
    _initial_terms_: List[LiteralString]
    """The list of initial terms used in the linear model. These terms 
    include the categorical features and continuous features up to the 
    specified interaction order."""
    _p_values: 'Series[float]'
    """The `_p_values` attribute is a Pandas Series that stores the 
    p-values for the features in the linear regression model. This 
    attribute is an implementation detail and is not part of the public 
    API."""
    _anova: DataFrame
    """The `_anova` attribute is a Pandas DataFrame that stores the 
    ANOVA table for the fitted linear regression model. This attribute 
    is an implementation detail and is not part of the public API.
    Get a ANOVA-table by calling the `anova()` method."""
    _effects: Series
    """The `_effects` attribute is a Pandas Series that stores the 
    effects (coefficients) of the features in the linear regression 
    model. This attribute is an implementation detail and is not part 
    of the public API."""
    _vif: DataFrame
    """The `_vif` attribute is a Pandas DataFrame that stores the 
    Variance Inflation Factors (VIFs), the Generalized Variance
    Inflation Factors (GVIFs) and its threshold for the features in the 
    linear regression model. This attribute is an implementation detail 
    and is not part of the public API. Get a VIF-table by calling
    the `variance_inflation_factor()` method."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            categorical: List[str],
            continuous: List[str] = [],
            alpha: float = 0.05,
            order: int = 1,
            skip_intercept_as_least: bool = True,
            generalized_vif: bool = True,
            encode_categoricals: bool = True) -> None:
        assert order >= 0 and isinstance(order, int), (
            'Interaction order must be a positive integer')
        self.target = target
        self.categorical = categorical
        self.continuous = continuous
        self.output_map = {target: 'y'}
        _categorical = [f'x{i}' for i in range(len(categorical))]
        _continuous = [f'e{i}' for i in range(len(continuous))]
        self.input_map = (
            {f: _f for f, _f in zip(categorical, _categorical)}
            | {c: _c for c, _c in zip(continuous, _continuous)})
        self.input_rmap = {v: k for k, v in self.input_map.items()}
        self.alpha = alpha
        self.skip_intercept_as_least = skip_intercept_as_least
        self.generalized_vif = generalized_vif
        self.exclude = set()
        self._model = None
        self.data = (source
            .copy()
            .rename(columns=self.input_map|self.output_map))
        if encode_categoricals:
            self.data[_categorical] = self.data[_categorical].astype('category')
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
        """Get the initial formula for the ANOVA model (read-only).
        
        The initial formula is constructed from the target variable and 
        the initial terms, which are the terms with an interaction order 
        less than or equal to the specified order.
        """
        initial_formula = (
            f'{self.output_map[self.target]} ~ '
            + ' + '.join(self._initial_terms_))
        return initial_formula
    
    @property
    def uncertainty(self) -> float:
        """Get uncertainty of the model as square root of MS_Residual
        (read-only)."""
        if self._anova.empty or 'MS' not in self._anova:
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
    
    @property
    def effect_threshold(self) -> float:
        """Calculates the threshold for the effect of adding a term to
        the model. The threshold is calculated as the inverse survival 
        function (inverse of sf) at the given alpha (read-only)."""
        return float(stats.t.isf(self.alpha, self.model.df_resid))
    
    @property
    def design_matrix(self) -> DataFrame:
        """Get the design matrix of the current fitted model
        (read-only)."""
        dm = pd.DataFrame(
            self.model.model.data.exog,
            columns=self.model.model.data.xnames)
        dm[self.model.model.data.ynames] = self.model.model.data.endog
        return dm
    
    def _reset_tables_(self) -> None:
        """Reset the anova table, the p_values and the effects."""
        self._anova = pd.DataFrame()
        self._effects = pd.Series()
        self._p_values = pd.Series()
        self._vif = pd.DataFrame()
    
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
        sep = '[T.' if '[T.' in term_name else '['
        split = term_name.split(sep)
        if len(split) > 2:
            split = [sep.join(split[:-1]), split[-1]]
        split[0] = self.input_rmap.get(split[0], split[0])
        return sep.join(split)

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
        effects are described as absolute number of the parameter 
        coefficients devided by its standard error."""
        if self._effects.empty:
            self._effects = terms_effect(self.model)
        effects = self._effects.copy().rename(index=self._term_map_)
        return effects

    def p_values(self) -> 'Series[float]':
        """Get P-value for significance of adding model terms using 
        anova typ III table for current model."""
        if self._p_values.empty:
            self._p_values = terms_probability(self.model)
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
        self.p_values()
        p_values = self._p_values.copy()
        has_intercept = ANOVA.INTERCEPT in p_values.index
        if any(p_values.isna()):
            self.effects()
            effects = self._effects.copy()
            if has_intercept and self.skip_intercept_as_least:
                effects = effects.drop(ANOVA.INTERCEPT)
            smallest = np.where(effects == effects.min())[0][-1]
            least = str(effects.index[smallest])
        else:
            if has_intercept and self.skip_intercept_as_least:
                p_values = p_values.drop(ANOVA.INTERCEPT)
            least = str(p_values.index[p_values.argmax()])
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
    
    def r2_pred(self) -> float | None:
        """Calculate the predicted R-squared (R2_pred) for the fitted 
        model.
        
        Returns
        -------
        float
            The predicted R-squared value for the fitted model.
        
        Notes
        -----
        The predicted R-squared is a measure of how well the model would 
        predict new observations. It is calculated as:
        
        R2_pred = 1 - (Sum of Squared Prediction Errors / Total Sum of Squares)
        
        Where the prediction errors are calculated as the residuals 
        divided by (1 - leverage), where leverage is the diagonal 
        elements of the projection matrix P = X(X'X)^(-1)X'.

        References
        ----------
        Calculations are made according to:
        https://support.minitab.com/de-de/minitab/help-and-how-to/statistical-modeling/doe/how-to/factorial/analyze-factorial-design/methods-and-formulas/goodness-of-fit-statistics/#press
        
        Further information about projection matrix (influence matrix or
        hat matrix) can be found in:
        https://en.wikipedia.org/wiki/Projection_matrix
        """
        X = self.model.model.data.exog
        y = self.model.model.data.endog
        try:
            P = np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T))
        except LinAlgError:
            return None
        
        ss_pred = np.sum(np.square(self.model.resid / (1 - np.diag(P))))
        ss_tot = np.sum(np.square(y - np.mean(y)))
        return 1 - ss_pred / ss_tot
    
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
            'hierarchical': self.is_hierarchical(),
            'least_term': self._convert_term_name_(self.least_term()),
            'p_least': self.p_values().max(),
            's': self.uncertainty,
            'aic': self.model.aic,
            'r2': self.model.rsquared,
            'r2_adj': self.model.rsquared_adj,
            'r2_pred': self.r2_pred()}
        return pd.DataFrame(data, index=[index])
    
    def summary(
            self, 
            anova_typ: Literal['', 'I', 'II', 'III', None] = None,
            vif: bool = True, **kwds
            ) -> Summary:
        """Generate a summary of the fitted model.

        Parameters
        ----------
        anova_typ: Literal['', 'I', 'II', 'III', None] , optional
            If not None, add an ANOVA table of provided type to the 
            summary, by default None.
        vif : bool, optional
            If True, variance inflation factors (VIF) are added to the 
            anova table. Will only be considered if anova_typ is not 
            None, by default True
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
    
    def eliminate(self, term: str) -> None:
        """Removes the given term from the model by adding it to the 
        `exclude` set.
        
        Parameters
        ----------
        term : str
            The term to be removed from the model.
        
        Raises
        ------
        AssertionError: 
            If the given term is not in the model.
        """ 
        term = ANOVA.SEP.join(map(
            lambda x: get_term_name(self.input_map.get(x, x)),
            term.split(ANOVA.SEP)))
        assert term in self._term_names_, f'Given term {term} is not in model'
        self.exclude.add(term)

    def recursive_feature_elimination(
            self, rsquared_max: float = 0.99, ensure_hierarchy: bool = True,
            **kwds) -> Generator[DataFrame, Any, None]:
        """Perform a recursive feature elimination on the fitted model.
        
        This function starts with the complete model and recursively 
        eliminates features based on their p-values, until only 
        significant features remain in the model. The function yields 
        the goodness-of-fit metrics at each step of the elimination
        process.
        
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
        
        Yields
        ------
        DataFrame
            The goodness-of-fit metrics at each step of the recursive 
            feature elimination.
        """
        self._model = None
        self.exclude = set()
        self.fit(**kwds)
        max_steps = len(self._term_names_)
        step = -1
        for step in range(max_steps):
            if self.has_insignificant_term(rsquared_max):
                self.eliminate(self.least_term())
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
            if typ not in ('I', 'II', 'III'):
                if not self.has_significant_interactions():
                    typ = 'II'
                elif self.is_hierarchical():
                    typ = 'III'
                else:
                    typ = 'I'
            self._anova = anova_table(self.model, typ=typ)
        
        if vif:
            self.variance_inflation_factor()
            idx = [i for i in self._vif.index if i in self._anova.index]
            self._anova.loc[idx, ANOVA.VIF] = self._vif.loc[idx, ANOVA.VIF]
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
    
    def variance_inflation_factor(self, threshold: int = 5) -> DataFrame:
        """Calculate the variance inflation factor (VIF) and the 
        generalized variance inflation factor (GVIF) for each predictor
        variable in the fitted model.
            
        This function takes a regression model as input and returns a 
        DataFrame containing the VIF, GVIF (= VIF^(1/2*dof)), threshold 
        for GVIF, collinearity status and calculation kind for each 
        predictor variable in the model. The VIF and GVIF are measures
        of multicollinearity, which can help identify variables that are
        highly correlated with each other.
            
        Parameters
        ----------
        trheshold : int, optional
            The threshold for deciding whether a predictor is collinear.
            Common values are 5 and 10. By default 5.
        
        Returns
        -------
        DataFrame
            A DataFrame containing the VIF, GVIF, threshold, collinearity
            status and performed method for each predictor variable in the 
            model.
        """
        if self._vif.empty:
            self._vif = variance_inflation_factor(
                self.model, threshold, generalized=self.generalized_vif)
        return self._vif.copy().rename(index=self._term_map_)

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
        data.name = ANOVA.RESIDUAL
        data.index.name = ANOVA.OBSERVATION
        data = data.to_frame().reset_index()
        data[ANOVA.PREDICTION] = self.model.predict()
        return data
    
    def _dfs_repr_(self) -> List[DataFrame]:
        """Returns a list of DataFrames containing the goodness-of-fit 
        metrics, ANOVA table, and parameter statistics for the fitted 
        model.
        
        Returns
        -------
        dfs : List[pandas.DataFrame]
            A list containing the following DataFrames:
            - Goodness-of-fit metrics
            - ANOVA table
            - Parameter statistics
            - VIF table (if possible)
        """
        if self.model is None:
            self.fit()
        vif = self.model.model.data.exog.shape[1] > 1
        dfs = [
            self.gof_metrics().drop('formula', axis=1),
            self.parameter_statistics(),
            self.anova(typ='I')]
        if vif:
            dfs.append(self.variance_inflation_factor())
        return dfs

    def _repr_html_(self) -> str:
        """Generates an HTML representation of the model's 
        goodness-of-fit metrics, ANOVA table, and parameter statistics.
        
        Returns
        -------
        str:
            An HTML-formatted string containing the model's diagnostic 
            information.
        """
        html = f'<b>{STR["formula"]}:</b></br>{self}</br></br>'
        html += frames_to_html(self._dfs_repr_(), STR['lm_repr_captions'])
        return html

    def __html__(self) -> str:
        """This method exists to inform other HTML-using modules (e.g. 
        Markupsafe, htmltag, etc) that this object is HTML and does not 
        need things like special characters (<>&) escaped."""
        return self._repr_html_()
    
    def __repr__(self) -> str:
        """Generates an string representation of the model's 
        goodness-of-fit metrics, ANOVA table, and parameter statistics.
        
        Returns
        -------
        str:
            An HTML-formatted string containing the model's diagnostic 
            information.
        """
        spacing = 2*'\n'
        _repr = f'{STR["formula"]}:\n{str(self)}'
        for df, caption in zip(self._dfs_repr_(), STR['lm_repr_captions']):
            _repr += spacing + caption + ':\n' + str(df)
        return _repr
    
    def __str__(self) -> str:
        """Generates a string representation of the linear regression 
        model's formula.
        
        The formula is constructed by iterating over the parameter 
        statistics and adding each parameter name and coefficient to the
        formula string. The target variable is included at the start of 
        the formula."""
        formula = f'{self.target} ~'
        param_stats = self.parameter_statistics()
        for name in param_stats.index:
            coef = float(param_stats.loc[name, 'coef']) # type: ignore
            if name == param_stats.index[0]:
                sign = '-' if coef < 0 else ''
            else:
                sign = '- ' if coef < 0 else '+ '
            _name = '' if name == ANOVA.INTERCEPT else f'*{name}'
            formula += f' {sign}{abs(coef):.4f}{_name}'
        return formula
