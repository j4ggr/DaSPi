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
import statsmodels.formula.api as smf

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


class LinearModel:
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
        'data', 'target', 'features', 'disturbances', '_model', '_alpha',
        'skip_intercept_as_least', 'generalized_vif', 'feature_map', 
        'main_term_map', 'target_map', 'excluded', '_initial_terms', 
        '_p_values', '_anova', '_effects', '_vif')
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
            encode_features: bool = True) -> None:
        assert order > 0 and isinstance(order, int), (
            'Interaction order must be a positive integer')
        for column in features + disturbances:
            assert column in source, f'Column {column} not found in source!'
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
            .copy()
            .rename(columns=self.feature_map | self.target_map))
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
            f'{self.target_map[self.target]} ~ '
            + ' + '.join(self._initial_terms))
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
    
    @property
    def _repr_captions(self) -> Tuple[str, str, str, str]:
        captions = (
                STR['lm_table_caption_summary'],
                STR['lm_table_caption_statistics'],
                STR['lm_table_caption_anova'],
                STR['lm_table_caption_vif'])
        return captions
    
    def _reset_tables_(self) -> None:
        """Reset the anova table, the p_values and the effects."""
        self._anova = pd.DataFrame()
        self._effects = pd.Series()
        self._p_values = pd.Series()
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
                'Unable to compute R2_pred.')
            return None
        except MemoryError as mem_err:
            warnings.warn(
                f'Memory error encountered: {mem_err}. '
                'Unable to compute R2_pred.')
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

        ``` python
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
            ).fit()
        dsp.ParameterRelevanceCharts(lm).plot().stripes().label()
        lm
        ```
        Now remove the least significant term from the model and refit.
        Repeat the process until the model contains only significant
        terms.
        
        ``` python
        lm.eliminate('stirrer:brand:catalyst').fit()
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

        ``` python
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
                f'Given parameter {parameter} was not excluded from model')
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

        ``` python
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
            ).fit()
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
            try:
                self._anova = anova_table(self.model, typ=typ)
            except ValueError:
                warnings.warn(
                    f"ANOVA table could not be computed with type {typ}."
                    " Using type III instead.",
                    RuntimeWarning)
                self._anova = anova_table(self.model, typ='III')

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
                        x = pd.Series(
                            RE.ENCODED_VALUE.findall(main_parameter),
                            dtype=self.data[term].dtype
                            )[0]
                    if x not in bounds.get(feature, (x,)):
                        continue

                else:
                    x_lower = self.data[term].min()
                    x_upper = self.data[term].max()
                    if parameter in bounds:
                        assert len(bounds[parameter]) == 2, (
                            f'Bounds for "{parameter}" must be a tuple of length 2.')
                        
                        _lower, _upper = sorted(bounds[parameter])
                        assert _lower >= x_lower and _upper <= x_upper, (
                            f'Bounds for "{parameter}" must be within the range of '
                            f'the data ({x_lower}, {x_upper}).')
                        
                        x_lower, x_upper = _lower, _upper

                    x = x_upper if maximize else x_lower
                
                xs_optimized[feature] = x
        return xs_optimized

    def residual_data(self) -> DataFrame:
        """
        Get the residual data from the fitted model.

        Returns
        -------
        pd.DataFrame
            The residual data containing the residuals, observation index, and predicted values.

        Examples
        --------
        
        ``` python
        import daspi
        df = daspi.load_dataset('partial_factorial')
        target = 'Yield'
        features = [c for c in df.columns if c != target]
        lm = LinearModel(df, target, features).fit()
        print(lm.residual_data())
        ```

        ```
            Observation      Residuals  Prediction
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
        ```
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
        html += frames_to_html(
            self._dfs_repr_(),
            captions=self._repr_captions)
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
        for df, caption in zip(self._dfs_repr_(), self._repr_captions):
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
