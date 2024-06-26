import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Any
from typing import Dict
from typing import Literal
from typing import Iterable
from numpy.typing import NDArray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .convert import get_term_name

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

def terms_effect(model: RegressionResultsWrapper) -> Series:
    """Calculates the impact of each term on the target. The
    effects are described as absolute number of the parameter 
    coefficients.

    Parameters
    ----------
    model : RegressionResultsWrapper
        Statsmodels regression results of fitted model.
    
    Returns
    -------
    Series
        A pandas Series containing the effects of each feature
        on the target variable.
    """
    params: Series = model.params
    names_map = {n: get_term_name(n) for n in params.index}
    effects = (params
        .abs()
        .rename(index=names_map)
        .groupby(level=0, axis=0)
        .sum()
        [uniques(names_map.values())])
    effects.name = ANOVA.EFFECTS
    effects.index.name = ANOVA.FEATURES
    return effects

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

def terms_probability(model: RegressionResultsWrapper) -> 'Series[float]':
    """Compute the p-values for the terms in a regression model using a
    ANOVA typ-III table.

    Parameters
    ----------
    model : RegressionResultsWrapper
        The regression model to compute the p-values for.

    Returns
    -------
    Series[float]
        A Series containing the p-values for each term in the model. If 
        the ANOVA table could not be calculated, the p-values will be
        set to NaN.
    
    Notes
    -----
    ANOVA typ III table is used, because it is the only one who gives us
    a p-value for the intercept.
    """
    anova = anova_table(model, typ='III')
    if anova.empty:
        names = model.model.data.design_info.term_names
        p_values = pd.Series({n: np.nan for n in names})
    else:
        p_values = anova['p'].iloc[:-1]
    return p_values


__all__ = [
    'uniques',
    'terms_effect',
    'variance_inflation_factor',
    'anova_table',
    'terms_probability',
]