import numpy as np

from math import exp
from typing import Any
from typing import Tuple
from typing import Sequence
from typing import Generator
from numpy.typing import NDArray
from pandas.core.series import Series

from scipy import stats
from scipy.stats import f
from scipy.stats import levene
from scipy.stats import ks_1samp
from scipy.stats import f_oneway
from scipy.stats import anderson
from scipy.stats import skewtest
from scipy.stats import ttest_ind
from scipy.stats import kurtosistest
from scipy.stats import fisher_exact
from scipy.stats import mannwhitneyu
from scipy.stats._distn_infrastructure import rv_continuous

from statsmodels.stats.proportion import test_proportions_2indep

from .._typing import NumericSample1D

# TDOD: further tests:
# from scipy.stats import chi2
# from scipy.stats import ansari
# from scipy.stats import kstest
# from scipy.stats import ranksums
# from scipy.stats import wilcoxon
# from statsmodels.stats.proportion import proportion_confint
# from statsmodels.stats.proportion import confint_proportions_2indep

def chunker(
        samples: Sequence[Any] | Series | NDArray,
        n_sections: int
        ) -> Generator[NDArray, Any, None]:
    """Divides the data into a specified number of sections.
    
    Parameters
    ----------
    sample : Sequence[Any]
        A one-dimensional array-like object containing the samples.
    n_sections : int
        Amount of sections to divide the data into.
        
    Yields
    ------
    NDArray
        A section of the data.
    
    Notes
    -----
    If equal-sized sections cannot be created, the first sections are 
    one larger than the rest.

    If more sections are to be created than the number of samples, 
    empty arrays are created.
    """
    assert n_sections > 0 and isinstance(n_sections, int)
    size, extras = divmod(len(samples), n_sections)
    sizes = extras*[size + 1] + (n_sections - extras)*[size]
    slicing_positions = np.array([0] + sizes).cumsum()

    _samples = np.asarray(samples)
    for i in range(n_sections):
        yield _samples[slicing_positions[i]:slicing_positions[i+1]]

def ensure_generic(
        dist: str | rv_continuous
        ) -> rv_continuous:
    """If the input is a string representing a distribution, convert it
    to a rv_continuous object.
    
    Parameters
    ----------
    dist : str or rv_continuous
        The distribution to convert. Can be either a string representing
        a distribution or a rv_continuous object.
    
    Returns
    -------
    rv_continuous
        The converted rv_continuous object if the input is a
        string representing a distribution, otherwise returns the input
        distribution directly.
    """
    if isinstance(dist, str):
        return getattr(stats, dist)
    else:
        return dist

def anderson_darling_test(
        sample: NumericSample1D
        ) -> Tuple[float, float]:
    """The Anderson-Darling test compares the measured values with the 
    theoretical values of a given distribution (in this case the normal 
    distribution). This test is considered to be one of the most 
    powerful tests for normal distribution for both small and large 
    sample sizes.

    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the samples.

    Returns
    -------
    p : float
        The p-value for the test
    A_star : float
        The adjusted Anderson Darling test statistic
    
    Notes
    -----
    This test was inspired by the Excel Addin by Charles Zaiontz, see:
    https://real-statistics.com/non-parametric-tests/goodness-of-fit-tests/anderson-darling-test/
    """
    N = len(sample)
    A2: float = anderson(sample, dist='norm')[0] # type: ignore
    A_star = A2*(1 + 0.75/N + 2.25/N**2)
    if 13 <= A_star:
        p = 0.0
    elif 0.6 <= A_star < 13:
        p = exp(1.2937 - 5.709*A_star + 0.0186*(A_star**2))
    elif 0.34 < A_star < 0.6:
        p = exp(0.9177 - 4.279*A_star - 1.38*(A_star**2))
    elif 0.2 < A_star <= 0.34:
        p = 1 - exp(42.796*A_star - 59.938*(A_star**2) - 8.318)
    else:
        p = 1 - exp(101.14*A_star - 223.73*(A_star**2) - 13.436)
    return p, A_star

def all_normal(
        *samples: NumericSample1D,
        p_threshold: float = 0.05
        ) -> bool:
    """Performs the Anderson-Darling test against the normal
    distribution for each given sample data. Only one-dimensional
    samples are accepted.
    
    Parameters
    ----------
    *samples : NumericSample1D
        One or more one-dimensional array-like objects containing the
        samples.
    p_threshold : float, optional
        The threshold p-value for significance (default is 0.05).
    
    Returns
    -------
    bool
        True if all p-values are greater than the specified p_threshold,
        False otherwise.
    
    Raises
    ------
    AssertionError
        If p_threshold is not within the range (0, 1).
    """
    assert 0 < p_threshold < 1, 'p_threshold must be within (0, 1)'
    return all([anderson_darling_test(x)[0] > p_threshold for x in samples])

def kolmogorov_smirnov_test(
        sample: NumericSample1D,
        dist: str | rv_continuous
        ) -> Tuple[float, float, Tuple[float, ...]]:
    """Perform a one-sample Kolmogorov-Smirnov-Test. This hypothesis
    test compares the underlying distribution F(x) of a sample against a 
    given distribution G(x). This test is valid only for continuous 
    distributions.
    
    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the samples.
    dist : str or scipy.stats rv_continous
        If a string, it should be the name of a continous distribution 
        in scipy.stats, which will be used as the cdf function.
    
    Returns
    -------
    p : float
        The two-tailed p-value for the test
    D : float
        Kolmogorov-Smirnov test statistic, either D, D+ or D-.
    params : Tuple[float, ...]
        Estimates for any shape parameters (if applicable), followed by 
        those for location and scale. For most random variables, shape 
        statistics will be returned, but there are exceptions 
        (e.g. ``norm``).
    """
    dist = ensure_generic(dist)
    params = dist.fit(sample)
    D, p = ks_1samp(
        sample, cdf=dist.cdf, args=params, alternative='two-sided')
    return p, D, params # type: ignore

def f_test(
        sample1: NumericSample1D,
        sample2: NumericSample1D
        ) -> Tuple[float, float]:
    """The F-test is a test for equal variances between two populations. 
    The probability distribution on which the F-test is based is called 
    the F-distribution (also Fisher distribution). 

    Parameters
    ----------
    sample1 : NumericSample1D
        A one-dimensional array-like object containing the first sample.
    sample2 : NumericSample1D
        A one-dimensional array-like object containing the second
        sample.

    Returns
    -------
    p : float
        The p-value for the test
    F : float
        The f-test statistic
    """
    F = float(np.var(sample1, ddof=1)/np.var(sample2, ddof=1))
    df1, df2 = len(sample1)-1, len(sample2)-1
    cumulated = float(f.cdf(F, df1, df2))
    p = 2 * min(cumulated, 1-cumulated)
    return p, F

def levene_test(
        sample1: NumericSample1D,
        sample2: NumericSample1D,
        heavy_tailed: bool = False
        ) -> Tuple[float, float]:
    """Perform Levene test for equal variances.
    The Levene test tests the null hypothesis that all input samples are 
    from populations with equal variances.
    
    Parameters
    ----------
    sample1 : NumericSample1D
        A one-dimensional array-like object containing the first sample.
    sample2 : NumericSample1D
        A one-dimensional array-like object containing the second
        sample.
    heavy_tailed : bool
        set True if data is heavy tailed, by default False 
    
    Returns
    -------
    p : float
        p-value for the test
    L : float
        Levene test statistic
    """
    center = 'trimmed' if heavy_tailed else 'median'
    L, p = levene(sample1, sample2, center=center)
    return p, L

def variance_stability_test(
        sample: NumericSample1D,
        n_sections: int = 3
        ) -> Tuple[float, float]:
    """Perform Levene test for equal variances within one sample.
    
    Divides the data into the number of n_sections. A Levene test is 
    then performed between these intercepts to check whether the 
    variance remains stable
    
    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the samples.
    n_sections : int, optional
        Amount of sections to divide the data into, by default 3

    Returns
    -------
    p : float
        p-value for the test
    L : float
        Levene test statistic
    """
    L, p = levene(*chunker(sample, n_sections), center='median')
    return p, L

def mean_stability_test(
        sample: NumericSample1D,
        n_sections: int = 3
        ) -> Tuple[float, float]:
    """Perform one-way ANOVA for equal means within one sample.
    
    Divides the data into the number of n_sections. A f_oneway test is 
    then performed between these intercepts to check whether the 
    mean remains stable
    
    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the samples.
    n_sections : int, optional
        Amount of sections to divide the data into, by default 3

    Returns
    -------
    p : float
        p-value for the test
    statistic : float
        The computed F statistic of the test.
    """
    statistic, p = f_oneway(*chunker(sample, n_sections))
    return p, statistic

def position_test(
        sample1: NumericSample1D,
        sample2: NumericSample1D,
        equal_var: bool = True,
        normal: bool | None = None,
        u_test: bool=True
        ) -> Tuple[float, float, str]:
    """calculate the test for the means of *two independent* samples of 
    scores.
    This is a two-sided test for the null hypothesis that 2 independent
    samples have identical average (expected) values. This test assumes
    that the populations have identical variances by default.
    If u_test is true and normal is false perform the Mann-Whitney U 
    rank test on two independent samples.
    The Mann-Whitney U test is a nonparametric test of the null 
    hypothesis that the distribution underlying sample x is the same as 
    the distribution underlying sample y. It is often used as a test of 
    difference in location between distributions.


    Parameters
    ----------
    sample1 : NumericSample1D
        A one-dimensional array-like object containing the first sample.
    sample2 : NumericSample1D
        A one-dimensional array-like object containing the second
        sample.
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample test 
        that assumes equal population variances. If False, perform 
        Welch's t-test, which does not assume equal population variance
    normal : bool or None
        Set to True if both sample data are normally distributed. If 
        True, perform a t-test. If False and u_test is True, perform a 
        Mann Whitney U test. If None, an Anderson-Darling test for 
        normal distribution is performed for both sample data. If one of
        the two data sets is not normally distributed, normal is set to 
        False, by default None
    u_test : bool, optional
        If True and data are not normally distributed, perform a Mann 
        Whitney U test, by default True

    Returns
    -------
    p : float
        p-value for the test
    statistic : float
        f if normal, else Levene test statistic
    test : string
        name of performed test
    """
    if not isinstance(normal, bool):
        normal = all_normal(sample1, sample2)

    if u_test and not normal:
        statistic, p = mannwhitneyu(
            sample1, sample2, alternative='two-sided', method='asymptotic')
        test = 'Mann-Whitney-U'
    else:
        statistic, p = ttest_ind(sample1, sample2, equal_var=equal_var)
        test = 't'
    return p, statistic, test # type: ignore

def variance_test(
        sample1: NumericSample1D,
        sample2: NumericSample1D,
        normal: bool | None = None,
        heavy_tailed: bool = False
        ) -> Tuple[float, float, str]:
    """Perform test for equal variances of two independent variables.
    This test tests the null hypothesis that all input samples are 
    from populations with equal variances.
    
    Parameters
    ----------
    sample1 : NumericSample1D
        A one-dimensional array-like object containing the first sample.
    sample2 : NumericSample1D
        A one-dimensional array-like object containing the second
        sample.
    normal : bool or None
        Set to True if both sample data are normally distributed. If 
        true, an F-test is performed, otherwise a Levene test. If None, 
        an Anderson-Darling test for normal distribution is performed 
        for both sample data. If one of the two data sets is not 
        normally distributed, normal is set to False, by default None
    heavy_tailed : bool
        set True if data is heavy tailed. Is only taken into account if 
        normal is False , by default False
    
    Returns
    -------
    p : float
        p-value for the test
    statistic : float
        f if normal, else Levene test statistic
    test : string
        name of performed test
    """
    if not isinstance(normal, bool):
        normal = all_normal(sample1, sample2)
    
    if normal:
        p, statistic = f_test(sample1, sample2)
        test = 'F'
    else:
        p, statistic = levene_test(sample1 ,sample2, heavy_tailed)
        test = 'Levenes'
        # statistic, p = ansari(sample1, sample2, alternative='two-sided')
    return p, statistic, test

def proportions_test(
        events1: int,
        observations1: int,
        events2: int,
        observations2: int,
        decision_threshold: int = 1000
        ) -> Tuple[float, float, str]:
    """Hypothesis test for comparing two independent proportions
    This assumes that we have two independent binomial samples.
    
    Fisher's exact test is one of exact tests. Especially when more than 
    20% of cells have expected frequencies < 5, we need to use Fisher's 
    exact test because applying approximation method is inadequate. 
    Fisher's exact test assesses the null hypothesis of independence 
    applying hypergeometric distribution of the numbers in the cells 
    of the table. 

    Parameters
    ----------
    events1 : int
        Counted number of events of sample 1.
    observations1 : int
        Total number of observations of sample 1.
    events2 : int
        counted number of events of sample 2.
    observations2 : int
        Total number of observations of sample 2.
    decision_threshold : int, optional
        if the sum of sample size (observations1 + observations2) is greater
        than decision_threshold, the Fisher exact test is performed, 
        by default 1000

    Returns
    -------
    p : float
        p-value for the test
    statistic : float
        test statistic
    test : string
        name of performed test
    """
    test = ''
    if observations1 + observations2 > decision_threshold:
        table = np.array([[events1, observations1], [events2, observations2]])
        statistic, p = fisher_exact(table, alternative='two-sided')
        test = 'Exakter Fisher'
    else:
        res = test_proportions_2indep(
            events1, observations1, events2, observations2, 
            method='wald', alternative='two-sided')
        p, statistic = res.pvalue, res.statistic
        test = 'Wald'
    return p, statistic, test # type: ignore

def kurtosis_test(
        sample: NumericSample1D
        ) -> Tuple[float, float]:
    """Two sided hypothesis test whether a dataset has normal kurtosis.

    This function tests the null hypothesis that the kurtosis of the 
    population from which the sample was drawn is that of the normal 
    distribution. Performs the calculations ignoring nan values
    
    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the samples.

    Returns
    -------
    p : float
        p-value for the test
    statistic : float
        The computed z-score for this test
    """
    statistic, p = kurtosistest(
        sample, nan_policy='omit', alternative='two-sided')
    return p, statistic

def skew_test(
        sample: NumericSample1D
        ) -> Tuple[float, float]:
    """Two sided hypothesis whether the skew is different from the 
    normal distribution.

    This function tests the null hypothesis that the skewness of the 
    population that the sample was drawn from is the same as that of a 
    corresponding normal distribution. Performs the calculations 
    ignoring nan values.
    
    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the samples.

    Returns
    -------
    p : float
        p-value for the test
    statistic : float
        The computed z-score for this test"""
    statistic, p = skewtest(sample, nan_policy='omit', alternative='two-sided')
    return p, statistic


__all__ = [
    'chunker',
    'ensure_generic',
    'anderson_darling_test',
    'all_normal',
    'kolmogorov_smirnov_test',
    'f_test',
    'levene_test',
    'variance_stability_test',
    'mean_stability_test',
    'position_test',
    'variance_test',
    'proportions_test',
    'kurtosis_test',
    'skew_test',]
