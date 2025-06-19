import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Any
from typing import Literal
from typing import Iterable
from collections import defaultdict
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
    ```python
    sequence = [1, 2, 3, 2, 1, 4, 5, 4]
    unique_elements = uniques(sequence)
    print(unique_elements)
    ```
    
    ```console
    [1, 2, 3, 4, 5]
    ```
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def terms_effect(model: RegressionResultsWrapper) -> Series:
    """Calculates the impact of each term on the target. The
    effects are described as absolute number of the parameter 
    coefficients devided by its standard error.

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
    se: Series = model.scale
    names_map = {n: get_term_name(n) for n in params.index}
    se = (model.bse
        .rename(index=names_map)
        .pow(2)
        .groupby(level=0, axis=0)
        .sum()
        .pow(1/2))
    params = (params
        .rename(index=names_map)
        .groupby(level=0, axis=0)
        .sum())
    effects = params.abs() / se
    effects = effects[uniques(names_map.values())]
    effects.name = ANOVA.EFFECTS
    effects.index.name = ANOVA.FEATURES
    return effects

def variance_inflation_factor(
        model: RegressionResultsWrapper,
        threshold: int = 5,
        generalized: bool = True) -> DataFrame:
    """Calculate the variance inflation factor (VIF) and the generalized
    variance inflation factor (GVIF) for each predictor variable in the
    fitted model.
        
    This function takes a regression model as input and returns a 
    DataFrame containing the VIF, GVIF (= VIF^(1/2*dof)), threshold for 
    GVIF, collinearity status and calculation kind for each predictor 
    variable in the model. The VIF and GVIF are measures of 
    multicollinearity, which can help identify variables that are highly
    correlated with each other.
        
    Parameters
    ----------
    model : RegressionResultsWrapper
        The regression model to analyze.
    threshold : int, optional
        The threshold for deciding whether a predictor is collinear.
        Common values are 5 and 10. By default 5.
    generalized : bool, optional
        Whether to calculate the generalized VIF or not, by default True.
    
    Returns
    -------
    DataFrame
        A DataFrame containing the VIF, GVIF, threshold, collinearity
        status and performed method for each predictor variable in the 
        model.
    
    Notes
    -----
    The VIF tells us: The degree to which the standard error of the 
    predictor is increased due to the predictor's correlation with the 
    other predictors in the model. VIF values greater than 10 
    (or, Tolerance values less than 0.10) corresponding to a multiple 
    correlation of 0.95 indicates a multicollinearity may be a problem 
    (Hair Jr, JF, Anderson, RE, Tatham, RL and Black, WC, 1998). Fox and 
    Weisberg also comment that the straightforward VIF can't be used if 
    there are variables with more than one degree of freedom (e.g. 
    polynomial and other contrasts relating to categorical variables 
    with more than two levels) and recommend using the gvif function 
    (generalized variance inflation factor) in the car package in R in 
    these cases. gvif is the square root of the VIF for individual 
    predictors and thus can be used equivalently. More generally 
    generalized variance-inflation factors consist of the VIF corrected 
    by the number of degrees of freedom (df) of the predictor variable: 
    GVIF = VIF[1/(2*df)] and may be compared to thresholds of 
    10[1/(2*df)] to assess collinearity using the stepVIF (source code: 
    https://github.com/cran/car/blob/master/R/vif.R) function in R.

    source: https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/Collinearity

    See Also:
    https://www.rdocumentation.org/packages/car/versions/3.1-2/topics/vif
    https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
    """
    exog = (pd
        .DataFrame(
            model.model.data.exog.copy(),
            columns=list(map(get_term_name, model.model.data.param_names)))
        .drop_duplicates()
        .reset_index(drop=True))
    assert exog.shape[1] > 1, (
        'To calculate VIFs, at least two predictor variables must be present.')
    
    terms = uniques(exog.columns)
    vifs = pd.DataFrame(index=terms, columns=ANOVA.VIF_COLNAMES)
    det = np.linalg.det
    interaction_order_terms = defaultdict(list)
    for term in terms:
        interaction_order_terms[len(term.split(ANOVA.SEP))].append(term)
    for order, _terms in interaction_order_terms.items():
        for term in _terms:
            Y = exog[term]
            XY = exog[_terms]
            if ANOVA.INTERCEPT in XY and term != ANOVA.INTERCEPT:
                XY = XY.drop(ANOVA.INTERCEPT, axis=1)
            X = XY.drop(term, axis=1)
            vif = None
            method = f'single_order-{order}_term'
            if X.empty:
                vif = 1.0
            elif Y.ndim == 2 and X.ndim == 2 and generalized:
                vif = (det(Y.corr()) * det(X.corr())) / det(XY.corr())
                method = 'generalized'
            if pd.isna(vif):
                y = Y.iloc[:, 0] if Y.ndim == 2 else Y
                X = sm.add_constant(X) if term != ANOVA.INTERCEPT else X
                vif = 1 / (1 - sm.OLS(y, X).fit().rsquared)
                method = 'R_squared'
            vifs.loc[term, ANOVA.VIF] = vif
            vifs.loc[term, 'Method'] = method

    vifs['DF'] = [exog[t].shape[1] if exog[t].ndim == 2 else 1 for t in terms]
    vifs['GVIF'] = vifs['VIF']**(1/(2*vifs['DF']))
    vifs['Threshold'] = threshold**(1/(2*vifs['DF']))
    vifs['Collinear'] = vifs['GVIF'] >= vifs['Threshold']
    return vifs

def anova_table(
        model: RegressionResultsWrapper,
        typ: Literal['I', 'II', 'III']
        ) -> DataFrame:
        """Perform an analysis of variance (ANOVA) on the fitted model.

        Parameters
        ----------
        model : RegressionResultsWrapper
            A fitted regression model of the `statsmodels` package.
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
        typs = ('I', 'II', 'III')
        assert typ in typs, f'typ must be one of {typs}'

        anova = pd.DataFrame()
        if all(model.pvalues.isna()):
            warnings.warn(
                'ANOVA table could not be calculated because the model is '
                'underdetermined.')
            return anova

        try:
            anova = sm.stats.anova_lm(model, typ=typ)
        except ValueError as error:
            for other_typ in (t for t in typs if t != typ):
                warnings.warn(
                    f'ANOVA table could not be computed with type {typ}. '
                    f'Using type {other_typ} instead.',
                    RuntimeWarning)
                try:
                    anova = sm.stats.anova_lm(model, typ=other_typ)
                except ValueError:
                    continue
                finally:
                    if anova.empty:
                        raise error
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