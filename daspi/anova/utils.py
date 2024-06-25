import warnings
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import Callable
from typing import Iterable
from numpy.typing import NDArray
from scipy.optimize import Bounds
from scipy.optimize import minimize
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.optimize._optimize import OptimizeResult
from statsmodels.regression.linear_model import RegressionResultsWrapper

from ..constants import RE
from ..constants import ANOVA


def uniques(seq: Iterable) -> list[Any]:
    """Get a list of unique elements from a sequence while preserving 
    the original order.

    Parameters
    ----------
    seq : Iterable
        The input sequence.

    Returns
    -------
    List[Any]
        A list of unique elements from the input sequence, preserving 
        the original order.

    Notes
    -----
    This function is based on the 'uniqify' algorithm by Peter Bengtsson.
    Source: https://www.peterbe.com/plog/uniqifiers-benchmark

    Examples
    --------
    >>> sequence = [1, 2, 3, 2, 1, 4, 5, 4]
    >>> unique_elements = preserved_order_uniques(sequence)
    >>> print(unique_elements)
    [1, 2, 3, 4, 5]
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_term_name(feature_name: str) -> str:
    """Get the original term name of a patsy encoded categorical
    feature, including interactions.

    Parameters
    ----------
    feature_name : str
        The encoded feature name.

    Returns
    -------
    str
        The original term name of the categorical feature.

    Notes
    -----
    Patsy encodes categorical features by appending '[T.<value>]' to the
    original term name. Interactions between features are represented by
    separating the feature names with ':'. This function extracts the
    original term name from the encoded feature name, taking into
    account interactions.

    Examples
    --------
    >>> encoded_name = 'Category[T.Value]:OtherCategory[T.OtherValue]'
    >>> term_name = get_term_name(encoded_name)
    >>> print(term_name)
    'Category:OtherCategory'
    """
    names = feature_name.split(ANOVA.SEP)
    matches = list(map(RE.ENCODED_NAME.findall, names))
    return ANOVA.SEP.join([(m[0] if m else n) for m, n in zip(matches, names)])

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

def is_main_feature(feature: str) -> bool:
    """Check if given feature is a main parameter (intercept is 
    excluded)."""
    return feature != ANOVA.INTERCEPT and ANOVA.SEP not in feature

def variance_inflation_factor(model: RegressionResultsWrapper) -> Series:
    """Calculate the variance inflation factor (VIF) for each predictor
    variable in the fitted model.

    Parameters
    ----------
    model : RegressionResultsWrapper
        Statsmodels regression results of fitted model. 

    Returns
    -------
    Series
        A pandas Series containing the VIF values for each predictor 
        variable.
    """
    xs: NDArray = model.model.data.exog.copy()
    assert xs.shape[1] > 1, (
        'To calculate VIFs, at least two predictor variables must be present.')
    _vif: Dict[str, float] = {}
    
    param_names = model.model.data.param_names
    names_map = {n: get_term_name(n) for n in param_names}
    for pos, name in enumerate(param_names):
        x = xs[:, pos]
        _xs = np.delete(xs, pos, axis=1)
        r2 = sm.OLS(x, _xs).fit().rsquared
        _vif[name] = (1 / (1 - r2))
    vif = pd.Series(_vif).rename(index=names_map)
    vif = vif[~vif.index.duplicated()]
    return vif

def anova_table(
        model: RegressionResultsWrapper,
        typ: Literal['I', 'II', 'III']
        ) -> DataFrame:
        """Perform an analysis of variance (ANOVA) on the fitted model.

        Parameters
        ----------
        typ : Literal['I', 'II', 'III'], optional
            The type of ANOVA to perform. Default is 'III', see notes
            for more informations about the types.
            - '' : If no or an invalid type is specified, Type-II is 
            used if the model has no significant interactions. 
            Otherwise, Type-III is used for hierarchical models and 
            Type-I is used for non-hierarchical models.
            - 'I' : Type I sum of squares ANOVA.
            - 'II' : Type II sum of squares ANOVA.
            - 'III' : Type III sum of squares ANOVA.

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
        Intercept. A discussion on which one to use can be  found here:
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
        """
        if all(model.pvalues.isna()):
            warnings.warn(
                'ANOVA table could not be calculated because the model is '
                'underdetermined.')
            return pd.DataFrame()

        anova = sm.stats.anova_lm(model, typ=typ)
        anova = anova.rename(
            columns={'df': 'DF', 'sum_sq': 'SS', 'PR(>F)': 'p'})

        anova['DF'] = anova['DF'].astype(int)
        anova['MS'] = anova['SS']/anova['DF']
        anova['n2'] = anova['SS'] / anova['SS'].sum()
        anova.index.name = ANOVA.SOURCE
        anova.columns.name = f'Typ-{typ}'
        return anova[ANOVA.TABLE_COLNAMES]

def frames_to_html(
        dfs: DataFrame | List[DataFrame] | Tuple[DataFrame, ...],
        captions: str | List[str] | Tuple[str, ...]) -> str:
    """Converts one or more DataFrames to HTML tables with captions.

    Parameters
    ----------
    dfs : DataFrame or list/tuple of DataFrames
        The DataFrame(s) to be converted to HTML.
    captions : str or list/tuple of str
        The captions to be used for the HTML tables. The number of 
        captions must match the number of DataFrames.

    Returns
    -------
    str
        The HTML representation of the DataFrames with captions.
    """
    if isinstance(captions, str):
        captions = (captions,)
    if isinstance(dfs, DataFrame):
        dfs = (dfs,)
    assert len(dfs) == len(captions), (
        "The number of DataFrames and captions must be equal.")
    spacing = 2*'</br>'
    html = ''
    for (df, caption) in zip(dfs, captions):
        if html:
            html += spacing
        html += (df
            .style
            .set_table_attributes("style='display:inline'")
            .set_caption(caption)
            .to_html())
    return html


# TODO: implement optimizer in model
# def optimize(
#         fun: Callable, x0: List[float], negate: bool, columns: List[str], 
#         mapper: dict, bounds: Bounds|None = None, **kwds
#         ) -> Tuple[List[float], float, OptimizeResult]:
#     """Base function for optimize output with scipy.optimize.minimize 
#     function

#     Parameters
#     ----------
#     fun : callable
#         The objective function to be minimized.
#     x0 : ndarray, shape (n,)
#         Initial guess. Array of real elements of size (n,), where n is 
#         the number of independent
#     negate : bool
#         If the function was created for maximization, the prediction is 
#         negated here again
#     mapper : dict or None
#         - key: str = feature name of main level
#         - value: dict = key: originals, value: codes
#     bounds : scipy optimizer Bounds
#         Bounds on variables
#     **kwds
#         Additional keyword arguments for `scipy.optimize.minimize`
#         function.
    
#     Returns
#     -------
#     xs : ndarray
#         Optimized values for independents
#     y : float
#         predicted output
#     res : OptimizeResult
#         The optimization result represented as a ``OptimizeResult`` object.
#         Important attributes are: ``x`` the solution array, ``success`` a
#         Boolean flag indicating if the optimizer exited successfully and
#         ``message`` which describes the cause of the termination. See
#         `OptimizeResult` for a description of other attributes.
#     """
#     if not bounds:
#         bounds = Bounds(-np.ones(len(x0)), np.ones(len(x0))) # type: ignore
#     res: OptimizeResult = minimize(fun, x0, bounds=bounds, **kwds)
#     xs = [decode(x, mapper, c) for x, c in zip(res.x, columns)]
#     y = -res.fun if negate else res.fun
#     return xs, y, res


__all__ = [
    'uniques',
    'get_term_name',
    'hierarchical',
    'is_main_feature',
    'variance_inflation_factor',
    'anova_table',
    'frames_to_html',
]
