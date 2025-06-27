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
- features: A list of column names in the DataFrame that represent 
  the exogenous (independent) variables.
- Additionally, it can take an optional input disturbances, which is a 
  list of column names representing exogenous variables that are logged 
  but cannot be influenced. No interactions are created for these
  variables.

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
import warnings
import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Set
from typing import Self
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import LiteralString
from typing import Generator
from patsy.desc import ModelDesc
from numpy.linalg import LinAlgError
from pandas.core.frame import DataFrame
from patsy.design_info import DesignInfo
from pandas.core.series import Series
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import forg
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.summary import summary_params_frame
from statsmodels.iolib.tableformatting import fmt_base
from statsmodels.regression.linear_model import RegressionResultsWrapper
from .convert import frames_to_html
from .convert import get_term_name

from .tables import anova_table
from .tables import terms_effect
from .tables import terms_probability
from .tables import variance_inflation_factor

from ..constants import RE
from ..constants import ANOVA

from ..strings import STR

from ..statistics.montecarlo import SpecLimits
from ..statistics.montecarlo import Specification
from ..statistics.estimation import GageEstimator
from ..statistics.estimation import MeasurementUncertainty


def is_main_parameter(parameter: str) -> bool:
    """Check if given parameter is a main parameter (intercept is 
    excluded)."""
    return parameter != ANOVA.INTERCEPT and ANOVA.SEP not in parameter

def get_order(parameter: str) -> int:
    """Get the order of a parameter (i.e., how many interactions it
    contains)."""
    return parameter.count(ANOVA.SEP) + 1

def hierarchical(parameters: List[str]) -> List[str]:
    """Get all parameters such that all lower interactions and main 
    effects are present with the same parameters that appear in the 
    higher interactions

    Parameters
    ----------
    parameters : list of str
        Columns of exogene variables
    
    Returns
    -------
    h_parameters : list of str
        Sorted features for hierarchical model"""

    h_parameters = set(parameters)
    for parameter in parameters:
        split = parameter.split(ANOVA.SEP)
        n_splits = len(split)
        for s in split:
            h_parameters.add(s)
        if n_splits <= ANOVA.SMALLEST_INTERACTION:
            continue

        for i in range(ANOVA.SMALLEST_INTERACTION, n_splits):
            for combo in map(ANOVA.SEP.join, itertools.combinations(split, i)):
                h_parameters.add(combo)

    return sorted(sorted(h_parameters), key=get_order)


class BaseHTMLReprModel(ABC):
    """Base class for HTML representation of models.
    
    This class provides a base implementation for generating HTML 
    representations of models. Those classes that inherit from this 
    class have an HTML representation that can be used for displaying 
    the model in a jupyter notebook or other HTML-based environment.
    """
    __slots__ = (
        '_captions', 'print_formula')
    _captions: Tuple[str, ...]
    target: str
    """The name of the target variable for the model."""
    print_formula: bool
    """Whether to print the formula in the HTML representation."""
    
    @property
    def captions(self) -> Tuple[str, ...]:
        """Get the captions for the tables used for html output
        (read-only)."""
        return self._captions
    
    @abstractmethod
    def _reset_tables_(self) -> None:
        """Reset the tables used for html output."""
        ...
    
    @abstractmethod
    def _dfs_repr_(self) -> List[DataFrame]:
        """Returns a list of DataFrames to be used for html output."""
        ...
    
    @abstractmethod
    def parameter_statistics(self) -> DataFrame:
        """Get the parameter statistics of the model.
        
        This method should return a DataFrame containing the parameter 
        statistics, such as coefficients, standard errors, t-values, 
        p-values, and confidence intervals for each parameter in the 
        model.
        
        Returns
        -------
        DataFrame:
            A DataFrame containing the parameter statistics.
        """
        ...

    def _repr_html_(self) -> str:
        """Generates an HTML representation of the model's 
        tables and its corresponding captions.
        
        Returns
        -------
        str:
            An HTML-formatted string containing the model's diagnostic 
            information.
        """
        html = frames_to_html(
            self._dfs_repr_(),
            captions=self.captions)
        if self.print_formula:
            html = f'<b>{STR["formula"]}:</b></br>{self}</br></br>{html}'
        return html

    def __html__(self) -> str:
        """This method exists to inform other HTML-using modules (e.g. 
        Markupsafe, htmltag, etc) that this object is HTML and does not 
        need things like special characters (<>&) escaped."""
        return self._repr_html_()
    
    def __repr__(self) -> str:
        """Generates an string representation of the model's 
        diagnostic information.
        
        Returns
        -------
        str:
            A string containing the model's diagnostic information.
        """
        spacing = 2*'\n'
        _repr = spacing.join(
            f'{c}:\n{df}' for df, c in zip(self._dfs_repr_(), self.captions))
        if self.print_formula:
           _repr = f'{STR["formula"]}:\n{str(self)}{spacing}{_repr}'
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


class LinearModel(BaseHTMLReprModel):
    """This class is used to create and simplify linear models so that 
    only significant features describe the model.
    
    Balanced models (DOEs or EVOPs) including continuous can be 
    analyzed. With this class, you can create an encoded design matrix
    with all factor levels, including their interactions. All
    non-significant factors can then be automatically eliminated.
    Furthermore, this class allows the examination of main effects, 
    the sum of squares (explained variation), and the ANOVA table in 
    more detail.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas DataFrame as tabular data in a long format used for the
        model.
    target : str
        Column name of the endogenous variable.
    features : List[str]
        Column names of the exogenous variables that can be actively 
        changed (factor levels for DOE or EVOP). Interactions are also 
        created for these variables if the order is set > 1.
    disturbances : List[str], optional
        Column names for exogenous variables that are logged but cannot 
        be influenced. No interactions are created for these variables.
    alpha : float, optional
        Threshold as alpha risk. All features, including continuous and 
        intercept, that have a p-value smaller than alpha are removed 
        during the automatic elimination of the factors. Default is 0.05.
    skip_intercept_as_least : bool, optional
        If True, the intercept is not removed as a least significant 
        term when using recursive feature elimination. Also if True, the
        intercept does not appear when calling the `least_parameter` method,
        by default True.
    encode_features : bool, optional
        If True, all of the provided feature variables are encoded using
        one-hot encoding by changing the data type to category. 
        Otherwise they are interpreted as continuous variables when 
        possible, by default True.
    fit_at_init : bool, optional
        If True, the model is fitted at initialization. Otherwise, the
        model is fitted after calling the `fit` method. Default is True.
    
    Examples
    --------
    Do some ANOVA and statistics on a dataset. Run the example below in 
    a Jupyther Notebook to see the results.


    ```python
    import daspi as dsp
    
    df = dsp.load_dataset('aspirin-dissolution')
    model = dsp.LinearModel(
        source=df,
        target='dissolution',
        features=['employee', 'stirrer', 'brand', 'catalyst', 'water'],
        disturbances=['temperature', 'preparation'],
        order=2)

    # Store goodnes of fit values for each elimination step
    df_gof = pd.concat(model.recursive_elimination())

    # Plot residual and parameter relevance analysis
    dsp.ResidualsCharts(model).plot().stripes().label(info=True)
    dsp.ParameterRelevanceCharts(model).plot().label(info=True)

    # Get HTML output
    model
    ```

    Notes
    -----
    Always be careful when removing the intercept. If the intercept is 
    missing, Patsy will automatically add all one-hot encoded levels for
    a categorical variable to compensate for this missing term. This 
    will result in extremely high VIFs. That means that the parameters
    are not linearly independent.

    Terminology
    -----------
    - feature: 
      The original name as it appears in the provided data.
    - term:
      The name of the term as it appears in the design matrix. All 
      feature names are converted to "xi" where i is an ascending 
      number. The name of the first feature becomes "x0", the second 
      becomes "x1",... The disturbance variables are also converted in 
      the same way, but instead of the letter "x" the letter "e" is 
      used.
    - parameter:
      The variable names in connection with a coefficient, but in the
      composition with the original feature names.
    """
    __slots__ = (
        'data',
        'target',
        'features',
        'disturbances',
        '_model',
        '_alpha',
        'skip_intercept_as_least',
        'generalized_vif',
        'feature_map', 
        'main_term_map',
        'target_map',
        'excluded',
        '_initial_terms', 
        '_p_values',
        '_anova',
        '_effects',
        '_vif')
    
    data: DataFrame
    """The Pandas DataFrame containing the data for the linear model."""
    target: str
    """The name of the target variable for the linear model."""
    features: List[str]
    """The list of features used in the linear model."""
    disturbances: List[str]
    """The list of disturbances variables used in the linear 
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
    feature_map: Dict[str, str]
    """A dictionary that maps the original feature names to the encoded 
    names (term) used in the model."""
    main_term_map: Dict[str, str]
    """A dictionary that maps the main term names (no interactions) back 
    to the original feature names used in the model."""
    target_map: Dict[str, str]
    """A dictionary that maps the original feature names to the encoded 
    feature names used in the model."""
    excluded: Set[str]
    """A set of feature names that should be excluded from the model."""
    _initial_terms: List[LiteralString]
    """The list of initial terms used in the linear model. These terms 
    include the encoded names of disturbances and the features with all 
    interactions up to the specified interaction order."""
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
            features: List[str],
            disturbances: List[str] = [],
            alpha: float = 0.05,
            order: int = 1,
            skip_intercept_as_least: bool = True,
            generalized_vif: bool = True,
            encode_features: bool = True,
            fit_at_init: bool = True
            ) -> None:
        assert order > 0 and isinstance(order, int), (
            'Interaction order must be a positive integer')
        
        for column in features + disturbances:
            assert column in source, f'Column {column} not found in source!'

        self._captions = (
                STR['lm_table_caption_summary'],
                STR['lm_table_caption_statistics'],
                STR['lm_table_caption_anova'],
                STR['lm_table_caption_vif'])
        
        self.target = target
        self.features = features
        self.disturbances = disturbances
        self.target_map = {target: 'y'}
        f_main_terms = [f'x{i}' for i in range(len(features))]
        d_main_terms = [f'e{i}' for i in range(len(disturbances))]
        self.feature_map = (
            {f: _f for f, _f in zip(features, f_main_terms)}
            | {c: _c for c, _c in zip(disturbances, d_main_terms)})
        self.main_term_map = {v: k for k, v in self.feature_map.items()}
        self.alpha = alpha
        self.skip_intercept_as_least = skip_intercept_as_least
        self.generalized_vif = generalized_vif
        self.excluded = set()
        self._model = None
        self.data = (source
            .rename(columns=self.feature_map | self.target_map)
            [list(self.feature_map.values()) + list(self.target_map.values())]
            .copy())
        
        if encode_features:
            self.data[f_main_terms] = self.data[f_main_terms].astype('category')
        model_desc = ModelDesc.from_formula(
            f'{self.target_map[self.target]}~'
            + ('*'.join(f_main_terms))
            + ('+'.join(['', *d_main_terms]) if d_main_terms else ''))
        terms = model_desc.describe().split(' ~ ')[1].split(' + ')
        self._initial_terms = [
            t for t in terms if t.count(ANOVA.SEP) < order]
        self._reset_tables_()
        
        if fit_at_init:
            self.fit()
        self.print_formula = True
    
    @property
    def fitted(self) -> bool:
        """Whether the model is fitted (read-only)."""
        return self._model is not None
        
    @property
    def model(self) -> RegressionResultsWrapper:
        """Get regression results of fitted model. Calls `fit` method
        if no model is fitted yet (read-only)."""
        if not self.fitted:
            warnings.warn('Model not fitted yet, calling `fit` method.')
            self.fit()
        return self._model # type: ignore

    @property
    def initial_formula(self) -> str:
        """Get the initial formula for the ANOVA model (read-only).
        
        The initial formula is constructed from the target variable and 
        the initial terms, which are the terms with an interaction order 
        less than or equal to the specified order.
        """
        initial_formula = (
            f'{self.target_map[self.target]} ~ '
            + ' + '.join(self._initial_terms))
        return initial_formula
    
    @property
    def uncertainty(self) -> float:
        """Get uncertainty of the model as square root of MS_Residual
        (read-only)."""
        if self._anova.empty or 'MS' not in self._anova:
            warnings.warn(
                'Could not calculate uncertainty from ANOVA table. '
                'ANOVA table is empty or MS_Residual not found.',
                UserWarning)
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
    def terms(self) -> List[str]:
        """Get the encoded names of all variables for the current 
        fitted model (read-only)."""
        return self.model.model.data.design_info.term_names
    
    @property
    def parameters(self) -> List[str]:
        """Get the names of all variables for the current fitted 
        model in the composition using the original feature and 
        disturbances names (read-only)."""
        return list(map(self._convert_term_name_, self.terms))

    @property
    def term_map(self) -> Dict[str, str]:
        """Get the names of all internal used and original terms 
        variables for the current fitted model as dict (read-only)"""
        return {n: self._convert_term_name_(n) for n in self.terms}
    
    @property
    def main_parameters(self) -> List[str]:
        """Get all main parameters of current model excluding intercept
        (read-only)."""
        main_parameters = [
            self.main_term_map[n] for n in self.terms if is_main_parameter(n)]
        return main_parameters
    
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
    
        ignore = list(self.excluded) + [ANOVA.INTERCEPT]
        terms = [t for t in self._initial_terms if t not in ignore]
        if ANOVA.INTERCEPT in self.excluded:
            terms = ['-1'] + terms
        return f'{self.target_map[self.target]} ~ {" + ".join(terms)}'
    
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
        self._effects = pd.Series(dtype='float64')
        self._p_values = pd.Series(dtype='float64')
        self._vif = pd.DataFrame()
    
    def _convert_single_term_name_(self, term_name: str) -> str:
        """Convert the single term name using the original names stored
        in `term_map`.

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
        split[0] = self.main_term_map.get(split[0], split[0])
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
        return all(term in self.terms for term in hierarchical(self.terms))
    
    def effects(self) -> Series:
        """Calculates the impact of each term on the target. The
        effects are described as absolute number of the parameter 
        coefficients devided by its standard error."""
        if self._effects.empty:
            self._effects = terms_effect(self.model)
        effects = self._effects.copy().rename(index=self.term_map)
        return effects

    def p_values(self) -> 'Series[float]':
        """Get P-value for significance of adding model terms using 
        anova typ III table for current model."""
        if self._p_values.empty:
            self._p_values = terms_probability(self.model)
        return self._p_values.copy().rename(index=self.term_map)

    def least_parameter(self) -> str:
        """Get the parameter name with the least effect or the least 
        p-value coming from a ANOVA typ III table of current fitted 
        model.

        Returns
        -------
        str
            The parameter name with the least p_value if possible
            or the parameter name with the least effect.

        Notes
        -----
        This method checks if any p-values are missing (NaN). If there
        are missing p-values, it returns the term name that has the 
        smallest effect on the target variable. Otherwise, it returns
        the term name with the least p-value for the F-stats coming from
        current ANOVA table.
        """
        p_values = self.p_values().copy()
        has_intercept = ANOVA.INTERCEPT in p_values.index
        if any(p_values.isna()):
            effects = self.effects().copy()
            if has_intercept and self.skip_intercept_as_least:
                effects = effects.drop(ANOVA.INTERCEPT)
            if any(effects.isna()):
                leasts = effects[effects.isna()]
            else:
                leasts = effects[effects == effects.min()]
            least = sorted(leasts.index, key=get_order)[-1]
        else:
            if has_intercept and self.skip_intercept_as_least:
                p_values = p_values.drop(ANOVA.INTERCEPT)
            least = p_values.iloc[::-1].idxmax()
        return str(least)
    
    def fit(self, **kwds) -> Self:
        """Create and fit a ordinary least squares model using current 
        formula. To fit with a user-defined formula, use the 
        `formula` keyword argument.
        
        Parameters
        ----------
        **kwds
            Pass formula and other keyword arguments to `ols` function
            of `statsmodels.formula.api`.
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
            P = X @ np.linalg.inv(X.T @ X) @ X.T
        except LinAlgError as linalg_err:
            warnings.warn(
                f'Linear algebra error encountered: {linalg_err}. '
                'Unable to compute R2_pred.',
                RuntimeWarning)
            return None
        except MemoryError as mem_err:
            warnings.warn(
                f'Memory error encountered: {mem_err}. '
                'Unable to compute R2_pred.',
                RuntimeWarning)
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
            recursive_elimination, the current step is passed as
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
            - 'least_parameter' = the least significant term
            - 'p_least' = The p-value of least significant term, coming
            from ANOVA table Type-III.
            - 'hierarchical' = True if model is hierarchical
        """
        self.anova(typ='III', vif=False)
        data = {
            'formula': self.formula,
            'hierarchical': self.is_hierarchical(),
            'least_parameter': self.least_parameter(),
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
            vif: bool = True,
            **kwds
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
    
    def eliminate(self, parameter: str) -> Self:
        """Removes the given parameter from the model by adding it to t
        he `exclude` set. Call `fit` to refit the model.
        
        Parameters
        ----------
        parameter : str
            The feature name, the disturbances name or the interaction 
            of multiple features to be removed from the model.
        
        Returns
        -------
        Self
            The current instance of the model for more method chaining.
        
        Examples
        --------
        Prepare a LinearModel instance, fit the model and plot the 
        relevance of the parameters. If you run the following code in
        a Jupyter Notebook, the plot and the html representation of the
        model will be displayed.

        ```python
        import pandas as pd
        import daspi as dsp

        df = dsp.load_dataset('aspirin-dissolution')
        lm = dsp.LinearModel(
                source=df,
                target='dissolution',
                features=['employee', 'stirrer', 'brand', 'catalyst', 'water'],
                disturbances=['temperature', 'preparation'],
                alpha=0.05,
                order=3,
                encode_categoricals=False
            )
        dsp.ParameterRelevanceCharts(lm).plot().stripes().label()
        lm
        ```
        Now remove the least significant term from the model and refit.
        Repeat the process until the model contains only significant
        terms.
        
        ```python
        lm.eliminate('stirrer:brand:catalyst')
        dsp.ParameterRelevanceCharts(lm).plot().stripes().label()
        lm
        ```

        To add again a feature to the model, use the `include` method.
        For an automatic elimination of insignificant terms, use the
        'recursive_elimination' method.

        Notes
        -----
        Always be careful when removing the intercept. If the intercept is 
        missing, Patsy will automatically add all one-hot encoded levels for
        a categorical variable to compensate for this missing term. This 
        will result in extremely high VIFs.

        Raises
        ------
        AssertionError: 
            If the given term is not in the model.
        """ 
        term = ANOVA.SEP.join(map(
            lambda x: get_term_name(self.feature_map.get(x, x)),
            parameter.split(ANOVA.SEP)))
        assert term in self.terms, (
            f'Given term {term} is not in model')
        self.excluded.add(term)
        return self

    def include(self, parameter: str) -> Self:
        """Adds the given feature to the model by removing it from the
        `excluded` set. Call `fit` to refit the model.

        Parameters
        ----------
        parameter : str
            The feature name, the disturbances name or the interaction 
            of multiple features to be added to the model.
        
        Returns
        -------
        Self
            The current instance of the model for more method chaining.
        
        Examples
        --------
        See `eliminate` method. After removing a term from the model,
        you can add it again by using the `include` method.

        ```python
        lm.include('C').fit()
        dsp.ParameterRelevanceCharts(lm).plot().stripes().label()
        lm
        ```
        """
        term = ANOVA.SEP.join(map(
            lambda x: get_term_name(self.feature_map.get(x, x)),
            parameter.split(ANOVA.SEP)))
        if term not in self.excluded:
            warnings.warn(
                f'Given parameter {parameter} was not excluded from model',
                UserWarning)
        self.excluded.discard(term)
        return self

    def recursive_elimination(
            self,
            rsquared_max: float = 0.99,
            ensure_hierarchy: bool = True,
            **kwds
            ) -> Generator[DataFrame, Any, None]:
        """Perform a recursive parameter elimination on the fitted model.
        
        This function starts with the complete model and recursively 
        eliminates parameters based on their p-values, until only 
        significant parameters remain in the model. The function yields 
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
        
        Examples
        --------

        Prepare a LinearModel instance, fit the model, automatically
        eliminate insignificant terms and plot the relevance of the
        parameters and the residuals. In the DataFrame df_gof you can 
        see when which feature was removed and the current 
        goodness-of-fit values can also be viewed. If you run the 
        following code in  a Jupyter Notebook, the plots and the html 
        representation of the model will be displayed.

        ```python
        import pandas as pd
        import daspi as dsp

        df = dsp.load_dataset('aspirin-dissolution')
        lm = dsp.LinearModel(
                source=df,
                target='dissolution',
                features=['employee', 'stirrer', 'brand', 'catalyst', 'water'],
                disturbances=['temperature', 'preparation'],
                alpha=0.05,
                order=3,
                encode_categoricals=False
            )
        df_gof = pd.concat(list(lm.recursive_elimination()))
        dsp.ParameterRelevanceCharts(lm).plot().stripes().label(info=True)
        dsp.ResidualsCharts(lm).plot().stripes().label(info=True)
        lm
        ```
        """
        self._model = None
        self.excluded = set()
        self.fit(**kwds)
        max_steps = len(self.terms)
        step = -1
        for step in range(max_steps):
            if self.has_insignificant_term(rsquared_max):
                self.eliminate(self.least_parameter())
                self.fit(**kwds)
                yield self.gof_metrics(step)
            else:
                break
        
        if step < 1:
            yield self.gof_metrics(step)
        
        if ensure_hierarchy and not self.is_hierarchical():
            step = step + 1
            h_parameters = hierarchical(self.terms)
            self.excluded = {e for e in self.excluded if e not in h_parameters}
            self.fit(**kwds)
            yield self.gof_metrics(step)
    
    def anova(
            self,
            typ: Literal['', 'I', 'II', 'III'] = '',
            vif: bool = False
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
        
        ```python
        import daspi as dsp
        df = dsp.load_dataset('anova3')
        lm = dsp.LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'])
        print(lm.anova(typ='III').round(3))
        ```

        ```console
        Typ-III    DF       SS       MS        F      p     n2
        source                                                
        Intercept   1  390.868  390.868  453.467  0.000  0.864
        Sex         1    2.075    2.075    2.407  0.127  0.005
        Risk        1   11.332   11.332   13.147  0.001  0.025
        Drug        2    0.816    0.408    0.473  0.626  0.002
        Residual   55   47.407    0.862      NaN    NaN  0.105
        ```
        """
        column_name = self._anova.columns.name
        if column_name and str(column_name).split('-')[1] != typ:
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
                use_t=use_t
            ).rename(columns={
                'P>|t|': 'p',
                'Conf. Int. Low': 'ci_low',
                'Conf. Int. Upp.': 'ci_upp'})
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
        threshold : int, optional
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
        return self._vif.copy().rename(index=self.term_map)

    def highest_parameters(
            self,
            features_only: bool = False) -> List[str]:
        """Determines all main and interaction parameters that do not 
        occur in a higher interaction in this constellation.
        
        Parameters
        ----------
        features_only : bool, optional
            If true, intercept and interaction parameters are not 
            returned. By default False.
        
        Returns
        -------
        List[str]
            List of highest parameters.
        """
        splitted_params = sorted(
            [p.split(ANOVA.SEP) for p in self.parameters], 
            key=len,
            reverse=True)
        
        _parameters: List[List[str]] = []
        highest_order = len(splitted_params[0])
        last_order = highest_order
        current_params = []
        for split in splitted_params:
            order = len(split)
            if order == highest_order:
                _parameters.append(split)
                continue

            if last_order != order and current_params:
                _parameters.extend(current_params.copy())
                current_params = []

            intersection_count = 0
            for parameter in _parameters:
                intersection_count += len(set(split) & set(parameter))
            if intersection_count < order:
                current_params.append(split)
            last_order = order
        
        parameters= [ANOVA.SEP.join(p) for p in _parameters + current_params]
        if features_only:
            ignore = self.disturbances + [ANOVA.INTERCEPT]
            parameters = [p for p in parameters if p not in ignore]
        return parameters
    
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
        n_terms_min = 2 if self.skip_intercept_as_least else 1
        if len(self.terms) == n_terms_min:
            return False
        
        if all(self.p_values().isna()):
            return True
        
        has_insignificant = (
            self._p_values.max() > self.alpha
            or self.model.rsquared > rsquared_max)
        return has_insignificant

    def has_significant_interactions(self) -> bool:
        """True if fitted model has significant interactions."""
        for parameter in self.highest_parameters():
            if ANOVA.SEP not in parameter:
                continue
            
            if self.p_values()[parameter] < self.alpha:
                return True
        return False
    
    @staticmethod
    def _as_tuple_(value: Any) -> Tuple:
        """Ensure that the given value is a tuple."""
        if isinstance(value, tuple):
            return value
        elif isinstance(value, (list, set)):
            return tuple(value)
        else:
            return (value,)

    def predict(self, xs: Dict[str, Any]) -> DataFrame:
        """Predict y with given xs. Ensure that all non interactions are 
        given in xs
        
        Parameters
        ----------
        xs : Dict[str, Any]
            The values for which you want to predict. Make sure that all
            non interaction parameters are given in xs. If multiple 
            values are to be predicted, provide a list of values for 
            each factor level.
            
        Returns
        -------
        DataFrame
            A DataFrame containing the predicted values for the given
            values of the predictor variables.
        """
        for parameter in self.main_parameters:
            assert parameter in xs, (
                f'Please provide a value for "{parameter}"')
        
        for x in xs.keys():
            assert x in self.main_parameters, (
                f'"{x}" is not a main parameter of the model.')
        
        xs = {
            self.feature_map[f]: self._as_tuple_(v) for f, v in xs.items()}
        df_pred = pd.DataFrame(xs)
        df_pred[self.target] = self.model.predict(df_pred)
        return df_pred.rename(columns=self.main_term_map)
    
    def optimize(
            self,
            maximize: bool = True,
            bounds: Dict[str, Any] = {}
            ) -> Dict[str, Any]:
        """Optimize the prediction by optimizing the parameters.

        Parameters
        ----------
        maximize : bool, optional
            Whether to maximize the prediction or minimize it.
        bounds : Dict[str, Any], optional
            Bounds for the parameters to optimize.
            - You can freeze a paramater by setting it to the desired 
                value. For example, to fix the value of a parameter to 1, 
                you can set bounds = {'param_name': 1}.
            - To keep an ordinal or metric parameter within a specific
                range, you can set bounds = {'param_name': (lower, upper)}.
                For example, to constrain a parameter to be between 0 and 
                1, you can set bounds = {'param_name': (0, 1)}.
            - In order to limit the selection of a nominal parameter 
                only to a certain subset of the originally conained values,
                you can set bounds = {'param_name': (val1, val2, ...)}.
                For example, to limit the selection of a nominal parameter
                to the values 'A' and 'B', you can set 
                bounds = {'param_name': ('A', 'B')}.

        Returns
        -------
        Dict[str, Any]
            The optimized parameters.
        """
        minimize = not maximize
        bounds = {k: self._as_tuple_(v) for k, v in bounds.items()}
        xs_optimized = {k: v[0] for k, v in bounds.items() if len(v) == 1}

        coefficients = (self
            .parameter_statistics()
            .drop(ANOVA.INTERCEPT, axis=0, errors='ignore')
            .sort_values(by='coef', ascending=not maximize)
            ['coef'])

        for parameter, coef in coefficients.items():
            for main_parameter in str(parameter).split(ANOVA.SEP):
                nominal = RE.ENCODED_NAME.findall(main_parameter)
                feature = nominal[0] if nominal else main_parameter
                term = self.feature_map[feature]

                if feature in xs_optimized:
                    continue

                if nominal:
                    if (maximize and coef < 0) or (minimize and coef > 0):
                        x = self.data[term].iloc[0]
                    else:
                        dtype = self.data[term].dtype
                        values = RE.ENCODED_VALUE.findall(main_parameter)
                        if dtype.name == 'category':
                            dtype = dtype.categories.dtype # type: ignore
                        x = pd.Series(values, dtype=dtype.name)[0]
                    if x not in bounds.get(feature, (x,)):
                        continue

                else:
                    x_lower = self.data[term].min()
                    x_upper = self.data[term].max()
                    if parameter in bounds:
                        _bounds = sorted(bounds[str(parameter)])
                        assert len(_bounds) == 2, (
                            f'Bounds for "{parameter}" must be a tuple of '
                            'length 2.')
                        
                        _lower, _upper = _bounds
                        assert _lower >= x_lower and _upper <= x_upper, (
                            f'Bounds for "{parameter}" must be within the '
                            f'range of the data ({x_lower}, {x_upper}).')
                        
                        x_lower, x_upper = _lower, _upper

                    x = x_upper if maximize else x_lower
                
                xs_optimized[feature] = x
        return xs_optimized

    def residual_data(self) -> DataFrame:
        """
        Get the residual data from the fitted model.

        Returns
        -------
        DataFrame
            The residual data containing the residuals, observation index, and predicted values.

        Examples
        --------
        
        ```python
        import daspi as dsp
        df = dsp.load_dataset('partial_factorial')
        target = 'Yield'
        features = [c for c in df.columns if c != target]
        lm = LinearModel(df, target, features)
        print(lm.residual_data())
        ```

        ```console
            Observation      Residuals  Prediction
        0             1  9.250000e+00       46.75
        1             2  2.000000e+00       51.00
        2             3 -1.050000e+01       73.50
        3             4 -2.500000e-01       65.25
        4             5 -1.421085e-14       53.00
        5             6  1.025000e+01       44.75
        6             7 -2.500000e-01       67.25
        7             8 -1.050000e+01       71.50
        8             9  3.750000e+00       65.25
        9            10 -1.200000e+01       57.00
        10           11 -1.500000e+00       79.50
        11           12  9.250000e+00       83.75
        12           13 -1.000000e+01       59.00
        13           14 -3.250000e+00       63.25
        14           15  9.250000e+00       85.75
        15           16  4.500000e+00       77.50
        ```
        """
        data = self.model.resid
        data.name = ANOVA.RESIDUAL
        data = data.to_frame()
        data[ANOVA.PREDICTION] = self.model.predict()
        data[ANOVA.OBSERVATION] = np.arange(len(data)) + 1
        return data[[ANOVA.OBSERVATION, ANOVA.RESIDUAL, ANOVA.PREDICTION]]
    
    def _dfs_repr_(self) -> List[DataFrame]:
        """Returns a list of DataFrames containing the goodness-of-fit 
        metrics, ANOVA table, and parameter statistics for the fitted 
        model.
        
        Returns
        -------
        dfs : List[pandas.DataFrame]
            A list containing the following DataFrames:
            - Goodness-of-fit metrics
            - Parameter statistics
            - ANOVA table
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


class GageStudyModel(LinearModel):
    """Calculates uncertainties for a measurement system (MSA Type 1 
    study), supporting one or multiple GageEstimator instances. If 
    multiple are provided, the uncertainty for linearity is also 
    calculated.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas DataFrame as tabular data in a long format used for the
        model.
    target : str
        Column name for source data holding the measurement values.
    reference : str
        Column name holding the reference values for the measured parts.
        This column is also used to identify which measured 
        values belong to which reference part. If the column has missing 
        values, the pandas method `ffill()` is used to ensure that the 
        column is filled. The uncertainty u_lin is determined if the 
        column contains several reference values, unless a known 
        measurement uncertainty is given with the argument `u_lin`.
    tolerance : float | SpecLimits | Specification
        The specification limits for the measurement system. This can 
        be a float, a `SpecLimits` instance or a `Specification` 
        instance. If a float is given, it is interpreted as the 
        tolerance (e.g., 0.1 for ±0.05).
    resolution : float | None
        The resolution of the measurement system. If None, the
        resolution is estimated from the data. If a float is given, it
        is interpreted as the resolution (e.g., 0.01 for a resolution of
        0.01).
    u_cal : MeasurementUncertainty | float
        The measurement uncertainty of the gage used to measure the 
        reference value. This parameter quantifies the uncertainty 
        associated with the measurement device itself, reflecting how 
        much the measured value could vary due to the inherent 
        limitations of the gage. If a float is specified, it is assumed 
        to be the expanded uncertainty with a coverage factor of 
        `k = 2`, which typically corresponds to a 95% confidence level. 
        Please note that the coverage factor of the calibration is 
        independent of the coverage factor selected here.
    u_bi : MeasurementUncertainty | None, optional
        The uncertainty for bias `u_BI` can be specified here if it is 
        known; otherwise, it will be determined from the data. This
        parameter represents the systematic error that may affect the 
        measurement results, indicating how much the measured values 
        deviate from the true value due to consistent inaccuracies.
        The default is None.
    u_lin : MeasurementUncertainty | None, optional
        The measurement uncertainty for linearity `u_LIN` is determined 
        if multiple reference values are provided. This parameter 
        assesses how well the measured values conform to a linear 
        relationship with the reference values. However, it will not be 
        determined if a known measurement uncertainty is provided here. 
        The default is None.
    u_rest : MeasurementUncertainty | None, optional
        The uncertainty for further uncertainties not covered by the 
        parameters above. This parameter accounts for all additional 
        sources of uncertainty that may affect the overall measurement 
        but are not specifically addressed by the default uncertainties 
        provided here. It provides a more comprehensive assessment of 
        measurement uncertainty by considering factors that may arise 
        from environmental conditions, operator influences, or other 
        unknown variables. If known, it can be specified here; 
        otherwise, the default value is None.
    k : int | float, optional
        The coverage factor for expanded uncertainty. It is used as a 
        multiplier to determine the expanded uncertainty based on the 
        standard uncertainty. The value of `k` is typically set to 
        reflect the desired confidence level in the  measurement 
        results. Default is 2, typical values are:
        - k=2 corresponds to a confidence interval of 95.45%
        - k=3 corresponds to a confidence interval of 99.73%
    tolerance_ratio : float, optional
        The ratio of the tolerance to the standard deviation of the
        measurement system. If the ratio is below this limit, the
        measurement system is considered unacceptable. Default is 0.2.
    q_ms_limit : float, optional
        The limit for the Q_MS value. If the Q_MS is below this limit,
        the measurement system is considered acceptable. 
        Default is 0.15.
    cg_limit : float, optional
        The limit for the Gage R&R study's Cg value. If the Cg value
        below this limit, the measurement system is considered
        unacceptable. Default is 1.33.
    cgk_limit : float, optional
        The limit for the Gage R&R study's Cgk value. If the Cgk value
        is below this limit, the measurement system is considered
        unacceptable. Default is 1.33.
    resolution_ratio_limit : float, optional
        The ratio of the resolution to the standard deviation of the
        measurement system. If the ratio is below this limit, the
        measurement system is considered unacceptable. Default is 0.05.
    bias_corrected : bool, optional
        Indicates whether the bias is corrected for the Gage R&R study. 
        If True, the bias is not included in the measurement uncertainty; 
        otherwise, it is included. Default is False.
    
    Examples
    --------
    Run the following command in a jupyter notebook to get the html
    output of `gage` or you can also use `print(repr(gage))` instead:

    ```python
    import daspi as dsp
    df = dsp.load_dataset('grnr_layer_thickness')
    gage = dsp.GageStudyModel(
        source=df,
        target='result_gage',
        reference='reference',
        u_cal=df['U_cal'][0],
        tolerance=df['tolerance'][0],
        resolution=df['resolution'][0],)
    chart = dsp.GageStudyCharts(
            gage, stretch_figsize=1.5
        ).plot(
        ).stripes(
        ).label(
        ) # .save('path/to/file.png')
    gage # or print(repr(gage))
    ```

    Notes
    -----
    If only one reference is provided, the uncertainty for linearity 
    (LIN) will be 0.0. If multiple references are provided, the 
    uncertainty for linearity will be calculated based on the 
    standard deviation of the biases of the GageEstimator instances.
    """
    __slots__ = (
        'source',
        'target',
        'reference',
        '_gage',
        '_ref_gages',
        '_n_references',
        '_references_analysis',
        '_capabilities',
        '_df_u',
        '_u_re',
        '_u_bi',
        '_u_lin',
        '_u_evr',
        '_u_rest',
        '_u_ms',
        '_k',
        '_bias',
        '_bias_corrected',
        '_q_ms_limit',
        '_alpha',
        '_T_min_UMS')
    
    source: DataFrame
    target: str
    reference: str
    _gage: GageEstimator
    _ref_gages: List[GageEstimator]
    _n_references: int
    _references_analysis: DataFrame
    _capabilities: DataFrame
    _df_u: DataFrame
    _u_re: MeasurementUncertainty | None
    _u_bi: MeasurementUncertainty | None
    _u_lin: MeasurementUncertainty | None
    _u_evr: MeasurementUncertainty | None
    _u_rest: MeasurementUncertainty | None
    _u_ms: MeasurementUncertainty | None
    _k: int | float
    _bias: float | None
    _bias_corrected: bool
    _q_ms_limit: float
    _alpha: float
    _T_min_UMS: float | None

    def __init__(
            self,
            source: DataFrame,
            target: str,
            reference: str,
            tolerance: float | SpecLimits | Specification,
            resolution: float | None,
            u_cal: MeasurementUncertainty | float,
            u_bi: MeasurementUncertainty | None = None,
            u_lin: MeasurementUncertainty | None = None,
            u_rest: MeasurementUncertainty | None = None,
            k: int | float = 2,
            tolerance_ratio: float = 0.2,
            q_ms_limit: float = 0.15,
            cg_limit: float = 1.33,
            cgk_limit: float = 1.33,
            resolution_ratio_limit: float = 0.05,
            alpha: float = 0.05,
            bias_corrected: bool = False,) -> None:

        self.target = target
        self.reference = reference
        source = source[[target, reference]].copy()
        source[reference] = source[reference].ffill()
        self._n_references = source[reference].nunique()
        self.source = source
        self._u_re = None
        self._u_bi = u_bi
        self._u_lin = u_lin
        self._u_evr = None
        self._u_rest = u_rest
        self._u_ms = None
        self._T_min_UMS = None
        assert 0 < q_ms_limit < 1, (
            f'q_ms_limit must be greater than 0 and less than 1, '
            f'but {q_ms_limit} was provided.')
        self._q_ms_limit = q_ms_limit
        self._bias = None

        super().__init__(
            source=source,
            target=target,
            features=[reference],
            disturbances=[],
            alpha=alpha,
            order=1,
            fit_at_init=True,)
        
        self._gage = GageEstimator(
            samples=source[target] - source[reference],
            reference=0.0,
            u_cal=u_cal,
            tolerance=tolerance,
            resolution=resolution,
            tolerance_ratio=tolerance_ratio,
            agreement=2*k,
            cg_limit=cg_limit,
            cgk_limit=cgk_limit,
            resolution_ratio_limit=resolution_ratio_limit)

        self._ref_gages = []
        self._alpha = alpha
        self.k = k
        self._bias_corrected = bias_corrected
        self._captions = (
            STR['lm_table_caption_ref_gages'],
            STR['lm_table_caption_capabilities'],
            STR['lm_table_caption_mp_uncertainty'],)
        self._reset_tables_()
        self.print_formula = False
    
    @staticmethod
    def from_gage_estimators(
            gages: GageEstimator | List[GageEstimator],
            k: int | float = 2,
            bias_corrected: bool = False) -> 'GageStudyModel':
        """Create a GageStudyModel from a list of GageEstimator 
        instances. This method is useful when you already have 
        GageEstimator instances and want to create a GageStudyModel 
        without needing to provide the source DataFrame and other 
        parameters again.
        
        Parameters
        ----------
        gages : GageEstimator | List[GageEstimator]
            A list of GageEstimator instances to create the model from.
        k : int | float, optional
            The coverage factor for expanded uncertainty. It is used as 
            a multiplier to determine the expanded uncertainty based on 
            the standard uncertainty. The value of `k` is typically set 
            to reflect the desired confidence level in the  measurement 
            results. Default is 2, typical values are:
            - k=2 corresponds to a confidence interval of 95.45%
            - k=3 corresponds to a confidence interval of 99.73%
        
        Returns
        -------
        GageStudyModel
            A new GageStudyModel instance created from the provided 
            GageEstimator instances.
        """
        if isinstance(gages, GageEstimator):
            gages = [gages]
        
        gage0 = gages[0]
        target = str(gage0.samples.name) or ANOVA.TARGET
        source = pd.DataFrame()
        for gage in gages:
            if not isinstance(gage, GageEstimator):
                raise TypeError(
                    f'Expected GageEstimator, got {type(gage).__name__}')
            data = gage.samples.to_frame(name=target)
            data[ANOVA.REFERENCE] = gage.reference
            source = pd.concat([source, data], axis=0, ignore_index=True)
        model = GageStudyModel(
            source=source,
            target=target,
            reference=ANOVA.REFERENCE,
            u_cal=max(g.u_cal for g in gages),
            tolerance=gage0.tolerance,
            resolution=gage0.resolution,
            cg_limit=gage0.cg_limit,
            cgk_limit=gage0.cgk_limit,
            tolerance_ratio=gage0.tolerance_ratio,
            resolution_ratio_limit=gage0.resolution_ratio_limit,
            k=k,
            bias_corrected=bias_corrected)
        model._ref_gages = gages
        return model
    
    @staticmethod
    def _ensure_tuple_(
            value: float | Tuple[float, ...],
            n_values: int
            ) -> Tuple[float, ...]:
        """Ensure that the given value is a tuple."""
        if isinstance(value, tuple):
            values = value
        elif isinstance(value, (list, set)):
            values = tuple(value)
        else:
            values = (value,) * n_values
        assert len(values) == n_values, (
            f'Expected {n_values} values, got {len(values)}. '
            f'Got {values=}')
        return values
    
    @property
    def gage(self) -> GageEstimator:
        """Returns the GageEstimator instances (read-only)."""
        return self._gage
    
    @property
    def ref_gages(self) -> List[GageEstimator]:
        """Returns a list of GageEstimator instances used in the model
        (read_only).

        If only one reference is used, it returns a list containing 
        one GageEstimator of that reference. If multiple references
        are used, it returns a list of GageEstimator instances for each
        reference part measured."""
        if not self._ref_gages:
            for reference, group in self.source.groupby(self.reference):
                gage = GageEstimator(
                    samples=group[self.target],
                    reference=float(reference), # type: ignore
                    u_cal=self.gage.u_cal,
                    tolerance=self.gage.tolerance,
                    resolution=self.gage.resolution,
                    tolerance_ratio=self.gage.tolerance_ratio,
                    agreement=self.gage.agreement,
                    cg_limit=self.gage.cg_limit,
                    cgk_limit=self.gage.cgk_limit,
                    resolution_ratio_limit=self.gage.resolution_ratio_limit,)
                self._ref_gages.append(gage)
        return self._ref_gages
    
    @property
    def bias_corrected(self) -> bool:
        """Whether the bias is corrected for the Gage R&R study. If 
        True, the bias itself is not included in the measurement 
        uncertainty; otherwise, it is."""
        return self._bias_corrected

    @property
    def bias(self) -> float:
        """Returns the bias of the Gage study (read-only)."""
        if self._bias is None:
            self._bias = max(g.bias for g in self.ref_gages)
        return self._bias

    @property
    def n_samples(self) -> int:
        """The number of samples used in the Gage study (read-only)."""
        return self.gage.n_filtered

    @property
    def n_references(self) -> int:
        """Get the amount of references used in the study (read-only)."""
        return self._n_references

    @property
    def n_replications(self) -> int:
        """Get the number of replications for each reference (read-only). """
        return self.n_samples // self.n_references

    @property
    def tolerance(self) -> float:
        """Returns the tolerance of the first GageEstimator instance 
        (read-only)."""
        return self.gage.tolerance
    
    @property
    def resolution(self) -> float:
        """Returns the resolution of the first GageEstimator instance
        (read-only)."""
        return self.gage.resolution

    @property
    def k(self) -> int | float:
        """Get the coverage factor `k` used in uncertainty 
        calculations.

        This property returns the coverage factor, which is a multiplier 
        used to determine the expanded uncertainty based on the standard 
        uncertainty. The value of `k` is typically set to reflect the 
        desired confidence level in the measurement results.

        Set the coverage factor with a positive number, typical values 
        are:
        - k=2 corresponds to a confidence interval of 95.45%
        - k=3 corresponds to a confidence interval of 99.73%
        """
        return self._k
    @k.setter
    def k(self, k: int | float) -> None:
        assert k > 0, f'k must be positive, typically 2 or 3. Got {k=}'
        self._k = k
        self._reset_tables_()
    
    @property
    def u_cal(self) -> MeasurementUncertainty:
        """The expanded uncertainty of the calibration."""
        return self.gage.u_cal
    
    @property
    def u_re(self) -> MeasurementUncertainty:
        """The uncertainty of the resolution of the testing system
        (read-only)."""
        if self._u_re is None:
            self._u_re = MeasurementUncertainty(
                error_limit=self.resolution / 2,
                distribution='rectangular',
                k=self.k)
        return self._u_re
    
    @property
    def u_bi(self) -> MeasurementUncertainty:
        """The uncertainty of the bias of the testing system (read-only)."""
        if self._u_bi is None:
            if self.bias_corrected:
                self._u_bi = MeasurementUncertainty(standard=0, k=self.k)
            else:
                self._u_bi = MeasurementUncertainty(
                    error_limit=abs(self.bias),
                    distribution='rectangular',
                    k=self.k)
        return self._u_bi

    @property
    def u_lin(self) -> MeasurementUncertainty:
        """The uncertainty of linearity of the measurement system 
        (read-only)."""
        if self._u_lin is None:
            if self.n_references > 2:
                y = self.source[self.target]
                x = self.source[self.reference]
                ss_lin = (
                    sum(sm.OLS(y, sm.add_constant(x)).fit().resid**2)
                    - self.anova()['SS']['Residual'])
                s_lin = (ss_lin / (self.n_references - 2))**0.5
            else:
                s_lin = 0
            self._u_lin = MeasurementUncertainty(standard=s_lin, k=self.k)
        return self._u_lin
    
    @property
    def u_evr(self) -> MeasurementUncertainty:
        """The uncertainty of the expanded variance ratio of the testing
        system (read-only)."""
        if self._u_evr is None:
            self._u_evr = MeasurementUncertainty(
                standard=self.anova()['MS']['Residual']**0.5,
                k=self.k)
        return self._u_evr
    
    @property
    def u_rest(self) -> MeasurementUncertainty:
        """The uncertainty of the repeatability of the testing system
        (read-only)."""
        if self._u_rest is None:
            self._u_rest = MeasurementUncertainty(standard=0, k=self.k)
        return self._u_rest
    
    @property
    def u_ms(self) -> MeasurementUncertainty:
        """The uncertainty of the measurement system (read-only)."""
        if self._u_ms is None:
            self._u_ms = self.u_cal.combine_with(
                max(self.u_re, self.u_evr),
                self.u_bi,
                self.u_lin,
                self.u_rest)
        return self._u_ms
    
    @property
    def q_ms_limit(self) -> float:
        """Get the provided limit for the Q_MS as float between 0 and 1
        (read-only)."""
        return self._q_ms_limit
    
    @property
    def T_min_UMS(self) -> float:
        """The minimum allowed tolerance for this testing system based
        on the uncertainty (read-only)."""
        if self._T_min_UMS is None:
            self._T_min_UMS = 2 * self.u_ms.expanded / self.q_ms_limit
        return self._T_min_UMS
    
    def references_analysis(self) -> DataFrame:
        """Returns a DataFrame with the analysis of the reference 
        parts. 
        
        The analysis includes the GageEstimator statistics for each 
        reference part, such as Cg, Cgk, resolution ratio, and bias.

        Returns
        -------
        DataFrame
            Analysis of the reference parts with GageEstimator 
            statistics.
        
        Examples
        --------

        ```python
        import daspi as dsp
        df = dsp.load_dataset('grnr_layer_thickness')
        gage = dsp.GageStudyModel(
            source=df,
            target='result_gage',
            reference='reference',
            u_cal=df['U_cal'][0],
            tolerance=df['tolerance'][0],
            resolution=df['resolution'][0],)
        print(gage.references_analysis())
        ```

        ```console
             Ref     mean     Bias         s      R
        1  0.101  0.10066 -0.00034  0.000688  0.003
        ```
        """
        if not self._references_analysis.empty:
            return self._references_analysis
        
        column_names = ANOVA.REFERENCE_ANALYSIS_COLNAMES
        self._references_analysis = pd.DataFrame({
            column_names[0]: [g.reference for g in self.ref_gages],
            column_names[1]: [g.mean for g in self.ref_gages],
            column_names[2]: [g.bias for g in self.ref_gages],
            column_names[3]: [g.std for g in self.ref_gages],
            column_names[4]: [g.R for g in self.ref_gages],},
            index=[i + 1 for i in range(len(self.ref_gages))],)
        
        return self._references_analysis

    def capabilities(self) -> DataFrame:
        """Returns a DataFrame with the capabilities of the measurement 
        system. 
        
        The capabilities include Cg, Cgk, and the ratio of resolution
        the .

        Returns
        -------
        DataFrame
            Capabilities for Cp, Cpk, Cg, Cgk and the ratio of 
            resolution to standard deviation.
        
        Examples
        --------

        ```python
        import daspi as dsp
        df = dsp.load_dataset('grnr_layer_thickness')
        gage = dsp.GageStudyModel(
            source=df,
            target='result_gage',
            reference='reference',
            u_cal=df['U_cal'][0],
            tolerance=df['tolerance'][0],
            resolution=df['resolution'][0],)
        print(gage.capabilities())
        ```

        ```console
                    Value  Limit  Capable     T_min
        Cg        2.179005   1.33     True  0.018311
        Cgk       1.932051   1.33     True  0.021711
        RE_ratio  0.033333   0.05     True  0.020000
        U_MS      0.096371   0.15     True  0.019274
        p_BI      0.001024   0.05    False       NaN
        ```
        """
        if not self._capabilities.empty:
            return self._capabilities
        
        g = self.gage
        q_ms = self.u_ms.quality_indicator(self.tolerance)
        self._capabilities  = pd.DataFrame({
            ANOVA.CAPABILITY_COLNAMES[0]: [
                g.cg,
                g.cgk,
                g.resolution_ratio,
                q_ms,
                g.p_bias],
            ANOVA.CAPABILITY_COLNAMES[1]: [
                g.cg_limit,
                g.cgk_limit,
                g.resolution_ratio_limit, 
                self.q_ms_limit,
                self._alpha],
            ANOVA.CAPABILITY_COLNAMES[2]: [
                None if g.cg is None else (g.cg >= g.cg_limit),
                g.cgk >= g.cgk_limit,
                g.resolution_ratio <= g.resolution_ratio_limit,
                q_ms <= self.q_ms_limit,
                g.p_bias >= self._alpha],
            ANOVA.CAPABILITY_COLNAMES[3]: [
                g.T_min_cg,
                g.T_min_cgk,
                g.T_min_res,
                self.T_min_UMS,
                None]},
            index=ANOVA.CAPABILITY_ROWS)

        return self._capabilities

    def uncertainties(self) -> DataFrame:
        """Returns a DataFrame with the uncertainties for the 
        measurement system.

        The table contains the following rows:

        - CAL: Calibration uncertainty
        - RE: Resolution uncertainty
        - BI: Bias uncertainty
        - LIN: Linearity uncertainty
        - EVR: Equipment Variation on the Reference
        - REST: Other known and provided uncertainty
        - MS: Measurement System uncertainty

        The table contains the following columns:

        - u: The measurement uncertainty for the respective components
        - U: The expanded uncertainty as k * u
        - Q: The Quality Indicator serves as a quality indicator for the 
          measurement process, reflecting how well the measurement 
          system performs in relation to the specified requirements and 
          tolerances.
        - rank: The rank of the uncertainty component, where a lower
          rank indicates a more significant contribution to the overall 
          uncertainty.

        Returns
        -------
        DataFrame
            Uncertainties for CAL, RE, BI, LIN, EVR, REST, MS.
        
        Examples
        --------

        ```python
        import daspi as dsp
        df = dsp.load_dataset('grnr_layer_thickness')
        gage = dsp.GageStudyModel(
            source=df,
            target='result_gage',
            reference='reference',
            u_cal=df['U_cal'][0],
            tolerance=df['tolerance'][0],
            resolution=df['resolution'][0],)
        print(gage.uncertainties())
        ```

        ```console
                     u         U         Q  rank
        CAL   0.000100  0.000200  0.013333   4.0
        RE    0.000289  0.000577  0.038490   2.0
        BI    0.000196  0.000393  0.026173   3.0
        LIN   0.000000  0.000000  0.000000   NaN
        EVR   0.000688  0.001377  0.091785   1.0
        REST  0.000000  0.000000  0.000000   NaN
        MS    0.000723  0.001446  0.096371   NaN
        ```
        """
        if not self._df_u.empty:
            return self._df_u
        
        _us = tuple(
            getattr(self, f'u_{r.lower()}') for r in ANOVA.UNCERTAINTY_ROWS_MS)
        cols=ANOVA.UNCERTAINTY_COLNAMES
        rows=ANOVA.UNCERTAINTY_ROWS_MS
        df_u = pd.DataFrame({
            cols[0]: [u.standard for u in _us],
            cols[1]: [u.expanded for u in _us],
            cols[2]: [u.quality_indicator(self.tolerance) for u in _us]},
            index=rows)
        mask = (df_u.index != rows[-1]) & (df_u[cols[0]] > 0) 
        df_u.loc[mask, cols[3]] = df_u.loc[mask, cols[0]].rank(ascending=False)
        self._df_u = df_u
        return self._df_u
    
    def anova(
            self,
            typ: Literal['', 'I', 'II', 'III'] = 'I',
            vif: bool = False) -> DataFrame:
        return super().anova(typ, vif)
    
    def _dfs_repr_(self) -> List[DataFrame]:
        dfs = [
            self.references_analysis(),
            self.capabilities(),
            self.uncertainties(),]
        return dfs
    
    def _reset_tables_(self) -> None:
        """Reset the internal tables for reference analysis,
        capabilities and uncertainties."""
        super()._reset_tables_()
        self._references_analysis = pd.DataFrame()
        self._capabilities = pd.DataFrame()
        self._df_u = pd.DataFrame()


class GageRnRModel(LinearModel):
    """A linear regression model for Gage Repeatability and 
    Reproducibility (Gage R&R) analysis.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas DataFrame as tabular data in a long format used for the
        model.
    target : str
        Column name for source data holding the measurement values.
    part : str
        Column name for the parts variable (unit under test).
    gage : GageStudyModel
        The provided GageStudyModel instance contains the measurement 
        system's statistics. This class corresponds to measurement 
        system analysis type 1. To calculate the uncertainty for 
        linearity, provide multiple reference parts in the
        `GageStudyModel` instance.
    u_av : str | None, optional
        Column name of the variable that identifies the operator for 
        type 2 Gage R&R. If the operator has no influence (MSA Typ III)
        and was therefore not included in the data acquisition, set
        `u_av` to `None`. In this case, it is recommended to record 
        multiple locations, multiple part holders, or other possible 
        changes to the measuring system and specify them in `u_gv`.
        Default is None.
    u_gv: str | None, optional
        Specifies the measurement uncertainty of gage ss, which 
        reflects the comparability of the measurement system. This 
        uncertainty is determined based on the data. To account for this 
        uncertainty, provide the column name containing the categorical 
        variables for the different measurement systems, test nests, or 
        similar. If not specified, the default value is 'None'.
    u_t: MeasurementUncertainty | str | None, optional
        Indicates the measurement uncertainty related to temperature 
        dependence. This uncertainty can be assessed using the data 
        (Method A) or specified directly if known (Method B).

        - Method A: Enter the column name containing the categorical 
          variables for different temperatures.
        - Method B: Specify a MeasurementUncertainty instance directly.

        If neither method is applied, the default value is 'None'.
    u_stab: str | None, optional
        Represents the measurement uncertainty related to the stability 
        of the measurement system. This uncertainty is determined from 
        the data. To include this uncertainty, enter the column name 
        containing the categorical variables for the relevant 
        conditions. If not specified, the default value is 'None'.
    u_obj: MeasurementUncertainty | str | None, optional
        Specifies the measurement uncertainty associated with the 
        inhomogeneity of the object. This uncertainty can be determined 
        from the data (Method A) or directly specified if known 
        (Method B).

        - Method A: Provide the column name containing the relevant 
          categorical variables for calculation from the data.
        - Method B: Specify a MeasurementUncertainty instance directly.

        If neither method is applied, the default value is 'None'.
    u_rest: MeasurementUncertainty | None, optional
        Accounts for uncertainties not covered by the parameters above. 
        This parameter considers additional sources of uncertainty that 
        may impact the overall measurement, such as environmental 
        conditions, operator influences, or other unknown variables. If 
        known, specify the uncertainty here; otherwise, the default 
        value is 'None'.
    fit_at_init : bool, optional
        If True, the model is fitted at initialization. Otherwise, the
        model is fitted after calling the `fit` method. Default is True.
    
    Examples
    --------
    If you run the following code in a Jupyter Notebook cell, all tables
    will be calculated and output in an HTML format:

    ```python
    import daspi as dsp
    df = dsp.load_dataset('grnr_layer_thickness')
    gage = dsp.GageStudyModel(
        source=df,
        target='result_gage',
        reference='reference',
        u_cal=df['U_cal'][0],
        tolerance=df['tolerance'][0],
        resolution=df['resolution'][0],)
    rnr_model = dsp.GageRnRModel(
        source=df,
        target='result_rnr',
        part='part',
        gage=gage,
        u_av='operator')
    rnr_model # or print(repr(rnr_model))
    ```
    
    References
    ----------
    The calculations are done based on the following references:
    
    [1] Dr. Bill McNeese, BPI Consulting, LLC (09.2012)
    https://www.spcforexcel.com/knowledge/measurement-systems-analysis-gage-rr/anova-gage-rr-part-2/
    
    [2] Minitab, LLC (2025)
    https://support.minitab.com/de-de/minitab/help-and-how-to/quality-and-process-improvement/measurement-system-analysis/how-to/gage-study/crossed-gage-r-r-study/methods-and-formulas/gage-r-r-table/
    
    [3] Curt Ronniger, Software für Statistik, Schulungen und Consulting (2025)
    https://www.versuchsmethoden.de/Mess-System-Analyse.pdf

    [4] VDA Band 5, Mess- und Prüfprozesse. Eignung, Planung und Management
    (Juli 2021) 3. überarbeitete Auflage
    """

    __slots__ = (
        'part',
        'has_operator',
        'u_map',
        'n_levels',
        '_n_samples',
        '_rnr',
        '_df_u',
        '_df_ump',
        '_df_ums',
        '_gage',
        '_k',
        '_u_evo',
        '_u_av',
        '_u_gv',
        '_u_ia',
        '_u_t',
        '_u_stab',
        '_u_obj',
        '_u_rest',
        '_u_mp',
        '_evaluate_ia',
        )
    
    part: str
    """Column name of the part (unit under test) variable."""
    has_operator: bool
    """Indicates whether the model has an operator variable."""
    u_map: Dict[str, str]
    """Dictionary for mapping original names of uncertainties in the 
    source data to the uncertainty abbreviations used in the model."""
    n_levels: 'Series[int]'
    """Series with the number of levels for each variable. The amount
    of replications are stored under equipment variation 'EV'."""
    _n_samples: int
    _rnr: DataFrame
    _df_u: DataFrame
    _df_ump: DataFrame
    _df_ums: DataFrame
    _gage: GageStudyModel
    _k: int | float
    _u_evo: MeasurementUncertainty | None
    _u_av: MeasurementUncertainty | str | None
    _u_gv: MeasurementUncertainty | str | None
    _u_ia: MeasurementUncertainty | None
    _u_t: MeasurementUncertainty | str | None
    _u_stab: MeasurementUncertainty | str | None
    _u_obj: MeasurementUncertainty | str | None
    _u_rest: MeasurementUncertainty | None
    _u_mp: MeasurementUncertainty | None
    _evaluate_ia: bool | Literal['auto']
    names_map: Dict[str, str]

    def __init__(
            self,
            source: DataFrame,
            target: str,
            part: str,
            gage: GageStudyModel,
            u_av: str | None = None,
            u_gv: str | None = None,
            u_t: MeasurementUncertainty | str | None = None,
            u_stab: str | None = None,
            u_obj: MeasurementUncertainty | str | None = None,
            u_rest: MeasurementUncertainty | None = None,
            fit_at_init: bool = True,
            ) -> None:
        assert u_av or u_gv, (
            'Either the u_av (operator) or the u_gv (gage variation) '
            'must be specified')
        
        self.part = part
        self.has_operator = bool(u_av)
        self._gage = gage
        self.k = gage.k
        self._u_evo = None
        self._u_av = u_av
        self._u_gv = u_gv
        self._u_ia = None
        self._u_t = u_t
        self._u_stab = u_stab
        self._u_obj = u_obj
        self._u_rest = u_rest
        self._u_mp = None
        self._evaluate_ia = 'auto'
        u_old = (part, u_av, u_gv, u_t, u_stab, u_obj)
        u_new = ('PV', 'AV', 'GV', 'T', 'STAB', 'OBJ')
        self.u_map = {
            old: new for old, new in zip(u_old, u_new) if isinstance(old, str)}
        features = list(self.u_map.keys())
        self._n_samples = source[target].notna().sum()
        self.n_levels = source[features].nunique().rename(self.u_map)
        self.n_levels[ANOVA.EV] = self.n_samples // np.prod(self.n_levels)
            
        super().__init__(
            source=source,
            target=target,
            features=features,
            order=2,
            fit_at_init=fit_at_init)
        self._captions = (
            STR['lm_table_caption_summary'],
            STR['lm_table_caption_anova'],
            STR['lm_table_caption_rnr'],
            STR['lm_table_caption_ms_uncertainty'],
            STR['lm_table_caption_mp_uncertainty'],)
        self._reset_tables_()
        self.print_formula = False
    
    @property
    def n_samples(self) -> int:
        """The number of samples used in the Gage R&R study (read-only)."""
        return self._n_samples
    
    @property
    def gage(self) -> GageStudyModel:
        """The provided GageModel instances contains the measurement 
        system's statistics. This classes corresponds to measurement 
        system analysis type 1 (read-only)."""
        return self._gage
    
    @property
    def df_u(self) -> DataFrame:
        """Get the data frame with the whole uncertainty budget as the
        combination of the data frames of MS and MP (`df_ums` and 
        `df_ump`). If the uncertainties are not calculated yet, the
        method `uncertainties()` is called (read-only)."""
        if self._df_u.empty:
            self.uncertainties()
        return self._df_u

    @property
    def df_ums(self) -> DataFrame:
        """Get the data frame with the measurement uncertainty of the 
        measurement system as gage study type 1. If the uncertainties 
        are not calculated yet, the method `uncertainties()` is called 
        (read-only)."""
        if self._df_ums.empty:
            self.uncertainties()
        return self._df_ums

    @property
    def df_ump(self) -> DataFrame:
        """Get the data frame with the measurement uncertainty of the 
        measurement process as gage study type 2 or 3. If the 
        uncertainties are not calculated yet, the method 
        `uncertainties()` is called (read-only)."""
        if self._df_ump.empty:
            self.uncertainties()
        return self._df_ump
    
    @property
    def k(self) -> int | float:
        """The coverage factor for the expanded measurement uncertainty."""
        return self._k
    @k.setter
    def k(self, k: int | float) -> None:
        assert k > 0, f'k must be greater than 0, got {k}'
        if hasattr(self, '_df_u') and k != getattr(self, '_k', 0):
            self._df_u = pd.DataFrame()
        self._k = k
    
    @property
    def interactions(self) -> List[str]:
        """Get the interaction parameter names if interaction is present 
        in the fitted model (read-only)."""
        return [p for p in self.parameters if ANOVA.SEP in p]
    
    @property
    def tolerance(self) -> float:
        """The tolerance of the specification (read-only)."""
        return self.gage.tolerance
    
    @property
    def u_evo(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the equipment variation 
        (repeatability) on the object (read-only)."""
        if self._u_evo is None:
            rnr = self.rnr(evaluate_ia=self._evaluate_ia)
            self._u_evo = MeasurementUncertainty(
                standard=rnr['s'][ANOVA.EV],
                k=self.k)
        return self._u_evo
    
    @property
    def u_av(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the appraiser ss 
        (comparability of operators) (read-only)."""
        if self._u_av is None or isinstance(self._u_av, str):
            rnr = self.rnr(evaluate_ia=self._evaluate_ia)
            self._u_av = MeasurementUncertainty(
                standard=rnr['s'].get(ANOVA.AV, 0),
                k=self.k)
        return self._u_av
    
    @property
    def u_gv(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the gage ss 
        (compareability of measurement system) (read-only)."""
        if self._u_gv is None:
            self._u_gv = MeasurementUncertainty(standard=0, k=self.k)
        elif isinstance(self._u_gv, str):
            rnr = self.rnr(evaluate_ia=self._evaluate_ia)
            self._u_gv = MeasurementUncertainty(
                standard=rnr['s'].get(ANOVA.GV, 0),
                k=self.k)
        return self._u_gv

    @property
    def u_ia(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the interaction 
        (read-only)."""
        if self._u_ia is None:
            rnr = self.rnr(evaluate_ia=self._evaluate_ia)
            self._u_ia = MeasurementUncertainty(
                standard=rnr['s'].get(ANOVA.IA, 0),
                k=self.k)
        return self._u_ia
    
    @property
    def u_t(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the temperature
        (read-only)."""
        if self._u_t is None:
            self._u_t = MeasurementUncertainty(standard=0, k=self.k)
        elif isinstance(self._u_t, str):
            rnr = self.rnr(evaluate_ia=self._evaluate_ia)
            self._u_t = MeasurementUncertainty(
                standard=rnr['s'].get('T', 0),
                k=self.k)
        return self._u_t

    @property
    def u_stab(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the stability over time
        (read-only)."""
        if self._u_stab is None:
            self._u_stab = MeasurementUncertainty(standard=0, k=self.k)
        elif isinstance(self._u_stab, str):
            rnr = self.rnr(evaluate_ia=self._evaluate_ia)
            self._u_stab = MeasurementUncertainty(
                standard=rnr['s'].get('STAB', 0),
                k=self.k)
        return self._u_stab
    
    @property
    def u_obj(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the inhomogeneity of the 
        object (read-only)."""
        if self._u_obj is None:
            self._u_obj = MeasurementUncertainty(standard=0, k=self.k)
        elif isinstance(self._u_obj, str):
            rnr = self.rnr(evaluate_ia=self._evaluate_ia)
            self._u_obj = MeasurementUncertainty(
                standard=rnr['s'].get('OBJ', 0),
                k=self.k)
        return self._u_obj
    
    @property
    def u_rest(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the remaining 
        (read-only)."""
        if self._u_rest is None:
            self._u_rest = MeasurementUncertainty(standard=0, k=self.k)
        return self._u_rest

    @property
    def u_ms(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the measurement system
        (read-only)."""
        return self.gage.u_ms
    
    @property
    def u_mp(self) -> MeasurementUncertainty:
        """Get the measurement uncertainty of the measurement process
        (read-only)."""
        if self._u_mp is None:
            self._u_mp = self.gage.u_cal.combine_with(
                max(self.gage.u_re, self.gage.u_evr, self.u_evo),
                self.gage.u_bi,
                self.gage.u_lin,
                self.gage.u_rest,
                self.u_av,
                self.u_gv,
                self.u_ia,
                self.u_t,
                self.u_obj,
                self.u_stab,
                self.u_rest,)
        return self._u_mp

    def anova(
            self,
            typ: Literal['', 'I', 'II', 'III'] = 'I',
            vif: bool = False) -> DataFrame:
        return super().anova(typ, vif)
    
    def _ems_divisor(self, name: str) -> int:
        """Get the divisor for the mean square estimation for the
        specified name."""
        return int(np.prod(self.n_levels[self.n_levels.index != name]))

    def rnr(
            self,
            evaluate_ia: bool | Literal['auto'] = 'auto'
            ) -> DataFrame:
        """Get the Gage R&R analysis results as table.
        
        The table contains the following rows:

        - R&R: The sum of repeatability and reproducibility.
        - EV: The repeatability component (Equipement Variation).
        - AV or GV: The reproducibility component 
          (Appriser or Gage Variation).
        - Part: The part-to-part component.
        - Total: The sum of the repeatability, reproducibility, 
          and part-to-part.

        The table contains the following columns:

        - MS: The variance estimation.
        - MS/Total: The estimated variance divided by the total variance.
        - s: The standard deviation as square root of the variance.
        - 6s: The 6 times the standard deviation.
        - 6s/Total: The 6 times the standard deviation divided by the 
            total 6 times the standard deviation.
        - 6s/Tolerance: The 6 times the standard deviation divided by 
            the tolerance.

        Parameters
        ----------
        evaluate_ia: bool | Literal['auto'] = 'auto'
            Whether to include the interaction term in the analysis.
            If 'auto', the interaction term is included if it is 
            significant. Otherwise, the interaction term is excluded and
            the model is refit without the interaction term.

        Returns
        -------
        DataFrame
            The Gage R&R analysis results as table.

        Examples
        --------

        ```python
        import daspi as dsp
        df = dsp.load_dataset('grnr_layer_thickness')
        gage = dsp.GageStudyModel(
            source=df,
            target='result_gage',
            reference='reference',
            u_cal=df['U_cal'][0],
            tolerance=df['tolerance'][0],
            resolution=df['resolution'][0],)
        rnr_model = dsp.GageRnRModel(
            source=df,
            target='result_rnr',
            part='part',
            gage=gage,
            u_av='operator')
        print(rnr_model.rnr())
        ```

        ```console
                         MS  MS/Total         s        6s  6s/Total  6s/Tolerance
        R&R    6.736111e-07   0.02248  0.000821  0.004924  0.149932      0.164148
        EV     6.736111e-07   0.02248  0.000821  0.004924  0.149932      0.164148
        AV     0.000000e+00   0.00000  0.000000  0.000000  0.000000      0.000000
        Part   2.929174e-05   0.97752  0.005412  0.032473  0.988696      1.082437
        Total  2.996535e-05   1.00000  0.005474  0.032844  1.000000      1.094812
        ```
        
        Notes
        -----
        The mean squarse used for the calculation of the variances are
        computed using the `anova` method. The variance components are 
        then calculated as follows:

        **With interaction term:**

        $$
        s^2_{EV} = MS_{EV} = MS_{residual}
        $$

        $$
        s^2_{operator} = \\frac{MS_{operator} - MS_{IA}}{n_{parts} n_{replication}}
        $$

        $$
        s^2_{IA} = \\frac{MS_{IA} - MS_{EV}}{n_{replication}}
        $$

        $$
        s^2_{Part} = \\frac{MS_{Part} - MS_{IA}}{n_{operator} n_{replication}}
        $$

        $$
        s^2_{AV} = s^2_{operator} + s^2_{IA}
        $$

        **Without interaction term:**

        $$
        s^2_{EV} = MS_{EV} = MS_{residual}
        $$

        $$
        s^2_{operator} = \\frac{MS_{operator} - MS_{EV}}{n_{parts} n_{replication}}
        $$

        $$
        s^2_{part} = \\frac{MS_{part} - MS_{EV}}{n_{operator} n_{replication}}
        $$

        **Variance summary:**

        $$
        s^2_{RnR} = s^2_{EV} + s^2_{AV}
        $$

        $$
        s^2_{total} = s^2_{RnR} + s^2_{part}
        $$
        """
        if not self._rnr.empty and self._evaluate_ia == evaluate_ia:
            return self._rnr
        
        if evaluate_ia == 'auto':
            evaluate_ia = self.has_significant_interactions()
        assert evaluate_ia in (True, False), (
            f'evaluate_ia must be True, False, or auto, got {evaluate_ia}')
        self._evaluate_ia = evaluate_ia

        idx_map = {ANOVA.RESIDUAL: ANOVA.EV} | self.u_map
        idx_rnr_sum = (
            [ANOVA.EV]
            + [n for n in self.u_map.values() if n != ANOVA.PV]
            + [ANOVA.IA])
        idx_order = [ANOVA.RNR] + idx_rnr_sum + [ANOVA.PV, ANOVA.TOTAL]
        columns = ANOVA.RNR_COLNAMES
        if not self._evaluate_ia:
            idx_rnr_sum.pop(idx_rnr_sum.index(ANOVA.IA))
            idx_order.pop(idx_order.index(ANOVA.IA))
            for parameter in self.parameters:
                if ANOVA.SEP in parameter:
                    self.eliminate(parameter)
            if self.excluded:
                self.fit()

        anova = self.anova().copy().rename(index=idx_map)
        ms = anova['MS']
        ems: Dict[str, float] = {ANOVA.EV: ms[ANOVA.EV]}
        if self._evaluate_ia:
            ms_iai = [ms[i] for i in self.interactions]
            _ms = sum(ms_iai)
            ems[ANOVA.IA] = sum(
                max(0, (msi - ms[ANOVA.EV]) / self.n_levels[ANOVA.EV])
                for msi in ms_iai)
        else:
            _ms = ms[ANOVA.EV]
        for name in self.u_map.values():
            ems[name] = (ms[name] - _ms) / self._ems_divisor(name)
        
        rnr = pd.DataFrame(
            columns=ANOVA.RNR_COLNAMES,
            index=idx_order)
        rnr[columns[0]] = pd.Series(ems).clip(lower=0)
        rnr.loc[ANOVA.TOTAL, :] = rnr.sum()
        rnr.loc[ANOVA.RNR, :] = rnr.loc[idx_rnr_sum, :].sum()
        rnr[columns[1]] = rnr[columns[0]] / rnr[columns[0]][ANOVA.TOTAL]
        rnr[columns[2]] = np.sqrt(rnr[columns[0]])
        rnr[columns[3]] = 6 * rnr[columns[2]]
        rnr[columns[4]] = rnr[columns[3]] / rnr[columns[3]][ANOVA.TOTAL]
        rnr[columns[5]] = rnr[columns[3]] / self.tolerance
        self._rnr = rnr
        return self._rnr
    
    def uncertainties(self) -> DataFrame:
        """The uncertainties of the measurement system.
        
        The table contains the following rows:

        - RE: Resolution uncertainty
        - BI: Bias uncertainty
        - EVR: Equipment variation on the Reference (repeatability) 
        - MS: Measurement System uncertainty
        - EVO: Equipement Variation on the Object(repeatability)
        - AV: Appraiser Variation uncertainty
        - IA: Interaction uncertainty
        - MP: Measurement Process (or Precision) uncertainty

        The table contains the following columns:

        - u: The measurement uncertainty for the respective components
        - U: The expanded uncertainty as k * u
        - Q: The Quality Indicator serves as a quality indicator for the 
          measurement process, reflecting how well the measurement 
          system performs in relation to the specified requirements and 
          tolerances.
        - rank: The rank of the uncertainty component, where a lower
          rank indicates a more significant contribution to the overall 
          uncertainty.

        Returns
        -------
        u : pandas.DataFrame
            The uncertainties of the measurement system.
        
        Examples
        --------

        ```python
        import daspi as dsp
        df = dsp.load_dataset('grnr_layer_thickness')
        gage = dsp.GageStudyModel(
            source=df,
            target='result_gage',
            reference='reference',
            u_cal=df['U_cal'][0],
            tolerance=df['tolerance'][0],
            resolution=df['resolution'][0],)
        rnr_model = dsp.GageRnRModel(
            source=df,
            target='result_rnr',
            part='part',
            gage=gage,
            u_av='operator')
        model.uncertainties()
        print(model.df_u)
        ```

        ```console
                        u         U         Q  rank
        CAL      0.000100  0.000200  0.013333   5.0
        RE       0.000289  0.000577  0.038490   3.0
        BI       0.000196  0.000393  0.026173   4.0
        LIN      0.000000  0.000000  0.000000   NaN
        EVR      0.000688  0.001377  0.091785   2.0
        MS_REST  0.000000  0.000000  0.000000   NaN
        MS       0.000723  0.001446  0.096371   NaN
        EVO      0.000821  0.001641  0.109432   1.0
        AV       0.000000  0.000000  0.000000   NaN
        GV       0.000000  0.000000  0.000000   NaN
        IA       0.000000  0.000000  0.000000   NaN
        T        0.000000  0.000000  0.000000   NaN
        STAB     0.000000  0.000000  0.000000   NaN
        OBJ      0.000000  0.000000  0.000000   NaN
        MP_REST  0.000000  0.000000  0.000000   NaN
        MP       0.000850  0.001700  0.113305   NaN
        ```
        """
        
        if not self._df_u.empty:
            return self._df_u

        df_ums = self.gage.uncertainties().rename(index={'REST': 'MS_REST'})
        
        cols=ANOVA.UNCERTAINTY_COLNAMES
        rows=ANOVA.UNCERTAINTY_ROWS_MP
        uncertainties = tuple(getattr(self, f'u_{r.lower()}') for r in rows)
        df_ump = pd.DataFrame({
            cols[0]: [u.standard for u in uncertainties],
            cols[1]: [u.expanded for u in uncertainties],
            cols[2]: [u.quality_indicator(self.tolerance) for u in uncertainties]},
            index=['MP_REST' if r == 'REST' else r for r in rows])
        
        df_u = pd.concat([df_ums, df_ump], axis=0)
        mask = (~df_u.index.isin(('MS', 'MP'))) & (df_u[cols[0]] > 0) 
        df_u.loc[mask, cols[3]] = df_u.loc[mask, cols[0]].rank(ascending=False)

        self._df_u = df_u
        self._df_ums = df_u.loc[df_ums.index, :].copy()
        self._df_ump = df_u.loc[df_ump.index, :].copy()
        return self._df_ump
    
    def _dfs_repr_(self) -> List[DataFrame]:
        """Returns a list of DataFrames containing the goodness-of-fit 
        metrics, ANOVA table, and parameter statistics for the fitted 
        model.
        
        Returns
        -------
        dfs : List[pandas.DataFrame]
            A list containing the following DataFrames:
            - Goodness-of-fit metrics
            - Parameter statistics
            - ANOVA table
            - R&R table
        """
        if self.model is None:
            self.fit()
        
        dfs = [
            self.gof_metrics().drop('formula', axis=1),
            self.anova(),
            self.rnr(),
            self.df_ums,
            self.df_ump]
        return dfs
    
    def _reset_tables_(self) -> None:
        """Reset the anova table, the p_values and the effects."""
        super()._reset_tables_()
        self._rnr = pd.DataFrame()
        self._df_u = pd.DataFrame()
        self._df_ump = pd.DataFrame()
        self._df_ums = pd.DataFrame()


__all__ = [
    'is_main_parameter',
    'get_order',
    'hierarchical',
    'LinearModel',
    'GageStudyModel',
    'GageRnRModel']