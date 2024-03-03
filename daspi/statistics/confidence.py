# source for ci: https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#conf_int_of_var
import scipy
import numpy as np
import pandas as pd

from math import sqrt
from numpy import ndarray
from typing import Tuple
from typing import Iterable
from typing import Callable
from scipy.stats import t
from scipy.stats import f
from scipy.stats import chi2
from scipy.stats import norm
from pandas.core.frame import DataFrame
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def sem(sample: Iterable, ddof: int = 1):
    """Calculate the standard error of the mean (or standard error of
    measurement) of the values in the input array.

    Parameters
    ----------
    sample : array like
        sample data, Only one-dimensional sample are accepted
    ddof : int, optional
        Delta degrees-of-freedom. How many degrees of freedom to adjust
        for bias in limited sample relative to the population estimate
        of variance, by default 1.

    Returns
    -------
    se : float
        The standard error of the mean in the sample.
    """
    se = np.std(sample, ddof=ddof) / np.sqrt(np.size(sample))
    return se

def mean_ci(
        sample: Iterable, level: float = 0.95, n_groups: int = 1
        ) -> Tuple[float, float, float]:
    """Two sided confidence interval for mean of data.

    Parameters
    ----------
    sample : array like
        sample data, Only one-dimensional sample are accepted
    level : float in (0, 1), optional
        confidence level, by default 0.95
    n_groups : int, optional
        Used for Bonferroni method. 
        Amount of groups to adjust the alpha risk within each group,
        that the total risk is not exceeded, by default 1

    Returns
    -------
    x_bar : float
        expected value
    ci_low, ci_upp : float
        lower and upper confidence level
    
    Notes
    -----
    The underlying `t.interval` function assumes that the data follows a
    t-distribution. Additionally, this method assumes that the sample is 
    representative of the population and that the data is independent 
    and identically distributed.
    """ 
    level = 1 - confidence_to_alpha(level, two_sided=False, n_groups=n_groups)  # adjust ci if Bonferroni method is performed
    se = sem(sample)
    dof = len(sample)-1
    x_bar = np.mean(sample)
    ci_low, ci_upp = t.interval(level, dof, loc=x_bar, scale=se)
    return x_bar, ci_low, ci_upp

def median_ci(
        sample: Iterable, level: float = 0.95, n_groups: int = 1
        ) -> Tuple[float, float, float]:
    """Two sided confidence interval for median of data

    Parameters
    ----------
    sample : array like
        sample data, Only one-dimensional sample are accepted
    level : float in (0, 1), optional
        confidence level, by default 0.95
    n_groups : int, optional
        Used for Bonferroni method. 
        Amount of groups to adjust the alpha risk within each group,
        that the total risk is not exceeded, by default 1

    Returns
    -------
    median : float
        median of data
    ci_low, ci_upp : float
        lower and upper confidence level
    
    Notes
    -----
    The underlying `t.interval` function assumes that the data follows a
    t-distribution. Additionally, this method assumes that the sample is 
    representative of the population and that the data is independent 
    and identically distributed.
    """
    level = 1 - confidence_to_alpha(level, two_sided=False, n_groups=n_groups)  # adjust ci if Bonferroni method is performed
    se = sem(sample)
    dof = len(sample) - 1
    median = np.median(sample)
    ci_low, ci_upp = t.interval(level, dof, loc=median, scale=se)
    return median, ci_low, ci_upp

def variance_ci(
        sample: Iterable, level: float = 0.95, n_groups: int = 1
        ) -> Tuple[float, float, float]:
    """Two sided confidence interval for variance of data

    Parameters
    ----------
    sample : array like
        sample data, Only one-dimensional sample are accepted
    level : float in (0, 1), optional
        confidence level, by default 0.95
    n_groups : int, optional
        Used for Bonferroni method. 
        Amount of groups to adjust the alpha risk within each group,
        that the total risk is not exceeded, by default 1

    Returns
    -------
    s2 : float
        variance of data
    ci_low, ci_upp : float
        lower and upper confidence level
    """
    alpha = confidence_to_alpha(level, two_sided=True, n_groups=n_groups)
    s2 = np.var(sample, ddof=1) # do not remove ddof=1, default is 0!
    dof = len(sample) - 1
    ci_upp = dof * s2 / chi2.ppf(alpha, dof)
    ci_low = dof * s2 / chi2.ppf(1 - alpha, dof)
    return s2, ci_low, ci_upp

def stdev_ci(
        sample: Iterable, level: float = 0.95, n_groups: int = 1
        ) -> Tuple[float, float, float]:
    """Two sided confidence interval for standard deviation of data

    Parameters
    ----------
    sample : array like
        sample data, Only one-dimensional sample are accepted
    level : float in (0, 1), optional
        confidence level, by default 0.95
    n_groups : int, optional
        Used for Bonferroni method. 
        Amount of groups to adjust the alpha risk within each group,
        that the total risk is not exceeded, by default 1

    Returns
    -------
    s : float
        variance of data
    ci_low, ci_upp : float
        lower and upper confidence level
    """
    return tuple(map(sqrt, variance_ci(sample, level, n_groups)))

def proportion_ci(
        count: int, nobs: int, level: float = 0.95, n_groups: int = 1
        ) -> Tuple[float, float, float]:
    """confidence interval for a binomial proportion with a asymptotic 
    normal approximation.

    Parameters
    ----------
    count : int
        number of events
    nobs : int
        total number of trials
    level : float in (0, 1), optional
        confidence level, default 0.95
    n_groups : int, optional
        Used for Bonferroni method. 
        Amount of groups to adjust the alpha risk within each group,
        that the total risk is not exceeded, by default 1

    Returns
    -------
    portion : float
        portion as count/nobs
    ci_low, ci_upp : float
        lower and upper confidence level with coverage approximately ci. 
    """
    alpha = confidence_to_alpha(level, two_sided=False, n_groups=n_groups)
    ci_low, ci_upp = proportion_confint(count, nobs, alpha, 'normal')
    portion = count/nobs
    return portion, ci_low, ci_upp

def bonferroni_ci(
        data: DataFrame, target: str, feature: str, level: float = 0.95, 
        ci_func: Callable = stdev_ci, n_groups: int | None = None, 
        name: str='midpoint') -> pd.DataFrame:
    """Calculate confidence interval after bonferroni correction.
    The Bonferroni correction is a method to adjust the significance 
    level alpha. This is always necessary if you carry out several 
    "multiple" tests. In this case, the probability of the type I error 
    for all tests together is no longer 5% (or 1%), but significantly 
    more. This means that the risk that you will receive at least one 
    significant result, even though there is no effect at all, is 
    significantly increased with multiple tests. This is also referred 
    to as alpha error accumulation or alpha inflation.

    Parameters
    ----------
    data : pandas.DataFrame
        data frame containing sample and feature data
    target : str
        name of target sample data column
    feature : str
        name of categorical feature. The confidence intervals are 
        calculated separately for these groups
    level : float in (0, 1), optional
        confidence level, default 0.95
    ci_func : {mean_ci, stdev_ci, variance_ci}
        function to calculate needed confidence interval that returns
        the values in order: midpoint, lower ci, upper ci
    n_groups : int, optional
        Used for Bonferroni correction. 
        Amount of groups to adjust the alpha risk within each group,
        that the total risk is not exceeded, 
        If none is given, it calculates the number based on the given 
        groups (ngroups attribute of groupby object), by default None
    name : str, optional
        name of midpoints, by default 'midpoint'

    Returns
    -------
    df : DataFrame
        data containing groups, midpoints and confidence limits
    """
    columns = [name, 'ci_low', 'ci_upp']
    if not isinstance(feature, list): feature = [feature]
    groups = data.groupby(feature)[target]
    n_groups = n_groups if isinstance(n_groups, int) else groups.ngroups
    df = pd.DataFrame(
            groups.agg(lambda x: ci_func(x, level, n_groups)).dropna().to_list(),
            columns = columns
        ).dropna()
    
    cat_values = [groups.indices.keys()]
    if len(feature) > 1: cat_values = [list(v) for v in cat_values[0]]
    df[feature] = cat_values
    
    return df

def delta_mean_ci(x1: Iterable, x2: Iterable, level: float = 0.95
        ) -> Tuple[float, float, float]:
    """two sided confidence interval for mean difference of two
    independent variables.

    Parameters
    ----------
    x1, x2 : array like
        sample data, Only one-dimensional sample are accepted
    ci : float in (0, 1), optional
        confidence level, by default 0.95

    Returns
    -------
    delta : float
        difference of means of data
    ci_low, ci_upp : float
        lower and upper confidence level
    """
    alpha = confidence_to_alpha(level, two_sided=True)
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    # pooled standard deviation:
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    dof = n1 + n2 - 2
    t_crit = t.ppf(1 - alpha, dof)
    m1, m2 = np.mean(x1), np.mean(x2)
    delta = m1 - m2
    ci_low = (m1 - m2) - t_crit * np.sqrt(1 / len(x1) + 1 / len(x2)) * s
    ci_upp = (m1 - m2) + t_crit * np.sqrt(1 / len(x1) + 1 / len(x2)) * s

    return delta, ci_low, ci_upp

def delta_variance_ci(x1: Iterable, x2: Iterable, level: float = 0.95
        ) -> Tuple[float, float, float]:
    """two sided confidence interval for variance difference of two
    independent variables.

    Parameters
    ----------
    x1, x2 : array like
        sample data, Only one-dimensional sample are accepted
    ci : float in (0, 1), optional
        confidence level, by default 0.95

    Returns
    -------
    delta : float
        difference of variance of data
    ci_low, ci_upp : float
        lower and upper confidence level
    
    Notes
    -----
    This function is a ChatGPT solution and therefore does not guarantee
    that this solution is correct.
    """
    alpha = confidence_to_alpha(level, two_sided=True)  
    s2_1, s2_2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    dof1, dof2 = len(x1) - 1, len(x2) - 1
    delta = s2_1 - s2_2
    F_upp = f.ppf(alpha, dof1, dof2)
    F_low = f.ppf(1 - alpha, dof1, dof2)
    ci_low = (s2_1 / s2_2) * F_low
    ci_upp = (s2_1 / s2_2) * F_upp
    return delta, ci_low, ci_upp

def delta_stdev_ci(x1: Iterable, x2: Iterable, level: float = 0.95
        ) -> Tuple[float, float, float]:
    """two sided confidence interval for standard deviation difference 
    of two independent variables.

    Parameters
    ----------
    x1, x2 : array like
        sample data, Only one-dimensional sample are accepted
    ci : float in (0, 1), optional
        confidence level, by default 0.95

    Returns
    -------
    delta : float
        difference of standard deviation of data
    ci_low, ci_upp : float
        lower and upper confidence level
    
    Notes
    -----
    This function is a ChatGPT solution and therefore does not guarantee
    that this solution is correct.
    """
    return tuple(map(sqrt, delta_variance_ci(x1, x2, level)))

def delta_proportions_ci(
        count1: int, nobs1: int, count2: int, nobs2: int,
        level: float = 0.95) -> Tuple[float, float, float]:
    """Confidence intervals for comparing two independent proportions
    This assumes that we have two independent binomial sample.


    Parameters
    ----------
    count1 : int
        number of events of sample 1
    nobs1 : int
        total number of trials of sample 1
    count2 : int
        number of events of sample 2
    nobs2 : int
        total number of trials of sample 2
    level : float in (0, 1), optional
        confidence level, by default 0.95

    Returns
    -------
    delta : float
        difference of variance of data
    ci_low, ci_upp : float
        lower and upper confidence level
    """
    alpha = confidence_to_alpha(level, two_sided=False)
    delta = count1/nobs1 - count2/nobs2
    ci_low, ci_upp = confint_proportions_2indep(
        count1, nobs1, count2, nobs2, 
        method='wald', compare='diff', alpha=alpha)
    
    return delta, ci_low, ci_upp

def fit_ci(
        results: RegressionResults, level: float = 0.95
        ) -> Tuple[ndarray, ndarray]:
    """calculate confidence interval fitted line. Applies to fitted WLS 
    and OLS models, not to general GLS
    
    Parameters
    ----------
    results : statsmodels RegressionResults
        fitted OLS or WLS model
    level : float in (0, 1), optional
        confidence level, by default 0.95
    
    Returns
    -------
    fit_ci_low, fit_ci_upp : numpy ndarray
        lower and upper confidence limits of fitted line
    
    Notes
    -----
    Using hat_matrix to calculate fit_se only works for fitted values

    This function is based on the summary_table function from the 
    statsmodels.stats.outliers_influence module, see: 
    https://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html
    """
    influence = OLSInfluence(results)
    alpha = confidence_to_alpha(level)
    tppf = t.isf(alpha, results.df_resid)
    fit_se = np.sqrt(influence.hat_matrix_diag * results.mse_resid)             # standard error for predicted mean (fitted line)
    fit_ci_low = results.fittedvalues - tppf * fit_se
    fit_ci_upp = results.fittedvalues + tppf * fit_se
    return fit_ci_low, fit_ci_upp

def prediction_ci(
        results: RegressionResults, level: float = 0.95
        ) -> Tuple[ndarray, ndarray]:
    """calculate confidence interval for prediction and to observe 
    outliers. Applies to fitted WLS and OLS models, not to general GLS.
    
    Parameters
    ----------
    results : statsmodels RegressionResults
        fitted OLS or WLS model
    level : float in (0, 1), optional
        confidence level, by default 0.95
    
    Returns
    -------
    pred_ci_low, pred_ci_upp : numpy ndarray
        lower and upper confidence limits of prediction
    """
    alpha = confidence_to_alpha(level)
    pred_ci_low, pred_ci_upp = wls_prediction_std(results, alpha=alpha)[1:]     # standard error for predicted observation
    return pred_ci_low, pred_ci_upp

def dist_prob_fit_ci(
        sample: Iterable, fit: Iterable, dist: str, level: float = 0.95
        ) -> Tuple[float, float]:
    """Calculate confidence interval for a fitting line when examining a
    distribution probability
    
    Parameters
    ----------
    sample : array like
        sample data
    fit : array like
        data points of fitting line
    dist : str or distribution generator of scipy package
        distribution being examined
    level : float in (0, 1), optional
        confidence level, by default 0.95
    
    Returns
    -------
    lower, upper : numpy ndarray
        lower and upper confidence limits of fitted line
    
    Notes
    -----
    Based on qqplot function of pinguin package:
    https://pingouin-stats.org/index.html
    """
    if isinstance(dist, str): 
        dist = getattr(scipy.stats, dist)
    alpha = confidence_to_alpha(level)
    sample = np.asarray(sample)
    fit = np.asarray(fit)
    sample.sort()
    fit.sort()
    n = len(sample)
    
    fit_params = dist.fit(sample)
    shape = fit_params[:-2] if len(fit_params) > 2 else None
    slope = (fit[-1] - fit[0]) / (sample[-1] - sample[0])
    
    P = _ppoints(n)
    crit = norm.ppf(1 - alpha)
    pdf = dist.pdf(sample) if shape is None else dist.pdf(sample, *shape)
    se = (slope / pdf) * np.sqrt(P * (1 - P) / n)
    upper = fit + crit * se
    lower = fit - crit * se
    return lower, upper

def confidence_to_alpha(
        confidence_level: float, two_sided: bool = True, n_groups: int = 1
        ) -> float:
    """Calculate significance level as alpha risk by given confidence 
    level
    
    Parameters
    ----------
    confidence_level : float in (0, 1)
        level of confidence interval
    two_sided : bool, optional
        True if alpha is to be calculated for a two-sided confidence
        interval, by default True
    n_groups : int
        Used for Bonferroni method. 
        Amount of groups to adjust the alpha risk within each group,
        that the total risk is not exceeded, by default 1
    
    Returns
    -------
    alpha : float
        significance level as alpha risk
    """
    assert 0 <= confidence_level <= 1, (
        f'Confidence level {confidence_level} not in (0, 1)')
    sides = 2 if two_sided else 1
    alpha = (1 - confidence_level)/(sides * n_groups)
    return alpha

def _ppoints(n: int, a: float=None) -> ndarray:
    """
    Ordinates For Probability Plotting.
    Numpy analogue of `R`'s `ppoints` function.

    Parameters
    ----------
    n : int
        Number of points generated
    a : float, optional
        Offset fraction (typically between 0 and 1), by default None

    Returns
    -------
    ppoints : ndarray
        Sequence of probabilities at which to evaluate the inverse
        distribution.
    
    Notes
    -----
    Based on _ppoints function of pinguin package:
    https://pingouin-stats.org/index.html
    """
    if not a:
        a = 3/8 if n <= 10 else 0.5
    return (np.arange(n) + 1 - a) / (n + 1 - 2*a)

__all__ = [
    mean_ci.__name__,
    median_ci.__name__,
    variance_ci.__name__,
    stdev_ci.__name__,
    proportion_ci.__name__,
    bonferroni_ci.__name__,
    delta_mean_ci.__name__,
    delta_variance_ci.__name__,
    delta_proportions_ci.__name__,
    fit_ci.__name__,
    prediction_ci.__name__,
    dist_prob_fit_ci.__name__,
    confidence_to_alpha.__name__,
    _ppoints.__name__]