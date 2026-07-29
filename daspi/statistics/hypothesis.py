"""Statistical hypothesis-testing functions.

This module collects a curated set of hypothesis tests commonly used in
industrial and scientific data analysis. The tests cover normality
checks, variance equality, location differences, distributional
goodness-of-fit, and shape statistics.

Every test function returns a tuple whose **first element is always the
p-value**, followed by the test statistic (and additional values where
applicable). This consistent signature makes it easy to pass the
functions to higher-level routines.

Normality tests
---------------
- `anderson_darling_test` – Anderson-Darling test against the normal
  distribution; recommended for both small and large samples.
- `all_normal` – convenience wrapper that checks whether *all* given
  samples pass the Anderson-Darling test.
- `kolmogorov_smirnov_test` – one-sample KS test against any
  continuous SciPy distribution.

Variance tests
--------------
- `f_test` – F-test for equal variances between two independent samples
  (assumes normality).
- `levene_test` – Levene test for equal variances; robust alternative to
  the F-test.
- `variance_stability_test` – internal variance stability of a single
  sample (Levene applied to time-ordered sections).
- `variance_test` – selects F-test or Levene test depending on
  normality.

Location / mean tests
---------------------
- `t_test` – one-sample t-test against a hypothesised population mean.
- `mean_stability_test` – internal mean stability of a single sample
  (one-way ANOVA applied to time-ordered sections).
- `position_test` – two-sample location test; dispatches to the
  independent-samples t-test or the Mann-Whitney U test based on
  normality and variance equality.

Proportion tests
----------------
- `proportions_test` – two-sample proportions test (automatically
  selects Fisher's exact test for small samples).

Shape tests
-----------
- `kurtosis_test` – D'Agostino kurtosis test.
- `skew_test` – D'Agostino skewness test.

Utilities
---------
- `chunker` – divides an array into *n* roughly equal sections; used
  internally by the stability tests.
- `ensure_generic` – normalises a distribution argument to a
  ``rv_continuous`` instance.

Notes
-----
The stability tests (`variance_stability_test`, `mean_stability_test`)
split a single time-ordered sample into sections and test whether the
statistic of interest is constant across sections — a lightweight
alternative to control-chart analysis.
"""
from collections.abc import Generator, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.core.series import Series
from scipy import stats
from scipy.stats import (
    anderson,
    f,
    f_oneway,
    fisher_exact,
    ks_1samp,
    kurtosistest,
    levene,
    mannwhitneyu,
    skewtest,
    ttest_1samp,
    ttest_ind,
)
from scipy.stats._distn_infrastructure import rv_continuous
from statsmodels.stats.proportion import test_proportions_2indep

from .._typing import NumericSample1D

__all__ = [
    'all_normal',
    'anderson_darling_test',
    'chunker',
    'dunn_test',
    'ensure_generic',
    'f_test',
    'kolmogorov_smirnov_test',
    'kurtosis_test',
    'levene_test',
    'mean_stability_test',
    'pairwise_tests',
    'position_test',
    'proportions_test',
    'skew_test',
    't_test',
    'variance_stability_test',
    'variance_test',
]

# TDOD: further tests:
# from scipy.stats import chi2
# from scipy.stats import ansari
# from scipy.stats import ranksums
# from scipy.stats import wilcoxon
# from statsmodels.stats.proportion import proportion_confint
# from statsmodels.stats.proportion import confint_proportions_2indep

def chunker(
        samples: Sequence[Any] | Series | NDArray,
        n_sections: int
        ) -> Generator[NDArray, Any]:
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
    
    Raises
    ------
    AssertionError
        If the input is a string that does not represent a valid 
        distribution in scipy.stats.
    """
    _dist: rv_continuous
    _dist = getattr(stats, dist) if isinstance(dist, str) else dist
    assert hasattr(stats, _dist.name), (
        f'{dist} is not a valid distribution')
    
    return _dist

def anderson_darling_test(
        sample: NumericSample1D
        ) -> tuple[float, float]:
    """The Anderson-Darling test compares the measured values with the
    theoretical values of a given distribution (in this case the normal
    distribution). This test is considered to be one of the most
    powerful tests for normality for both small and large sample sizes.

    The Anderson-Darling statistic $A^2$ is computed as a squared
    distance between the empirical cumulative distribution function and
    the theoretical CDF $F$, with stronger weighting in the tails. For
    a sample of size $N$ with ordered values $x_1 \\le \\ldots \\le x_N$:

    $$
        A^2 = -N - \\frac{1}{N} \\sum_{i=1}^{N} (2i-1)
              \\bigl(\\ln F(x_i) + \\ln(1 - F(x_{N-i+1}))\\bigr)
    $$

    To obtain a p-value, the statistic is adjusted for finite sample
    size:

    $$
        A^* = A^2 \\left(1 + \\frac{0.75}{N} + \\frac{2.25}{N^2}\\right)
    $$

    The p-value is computed by SciPy using interpolation from
    pre-calculated tables, which provides more accurate results than
    piecewise approximation formulas.

    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the samples.

    Returns
    -------
    p : float
        The p-value for the test.
    A_star : float
        The adjusted Anderson-Darling test statistic $A^*$.

    Notes
    -----
    This implementation uses SciPy's `anderson` function with
    `method='interpolate'` to compute accurate p-values from
    pre-calculated tables. The adjusted statistic $A^*$ is returned
    for compatibility with the literature and for manual interpretation
    using critical value tables.
    """
    N = len(sample)
    result = anderson(sample, dist='norm', method='interpolate')
    A2: float = result.statistic  # pyright: ignore[reportAttributeAccessIssue]
    p: float = result.pvalue  # pyright: ignore[reportAttributeAccessIssue]
    A_star = A2 * (1 + 0.75/N + 2.25/N**2)
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
    return all(anderson_darling_test(x)[0] > p_threshold for x in samples)

def kolmogorov_smirnov_test(
        sample: NumericSample1D,
        dist: str | rv_continuous,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
        ) -> tuple[float, float, tuple[float, ...]]:
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
    alternative : {'two-sided', 'less', 'greater'}, optional
        The alternative hypothesis to use for the test.
        'two-sided' : the cdf of the distribution is not the same as the 
            cdf of the sample
        'less' : the cdf of the distribution is the same as or less than 
            the cdf of the sample
        'greater' : the cdf of the distribution is the same as or greater 
            than the cdf of the sample
    
    Returns
    -------
    p : float
        The two-tailed p-value for the test
    D : float
        Kolmogorov-Smirnov test statistic, either D, D+ or D-.
    params : tuple[float, ...]
        Estimates for any shape parameters (if applicable), followed by 
        those for location and scale. For most random variables, shape 
        statistics will be returned, but there are exceptions 
        (e.g. ``norm``).
    """
    dist = ensure_generic(dist)
    params = dist.fit(sample)
    D, p = ks_1samp(
        sample, cdf=dist.cdf, args=params, alternative=alternative)
    return p, D, params # type: ignore

def f_test(
        sample1: NumericSample1D,
        sample2: NumericSample1D
        ) -> tuple[float, float]:
    """F-test for equal variances between two independent populations.

    The F-test compares the variances of two samples. The underlying
    probability distribution is the F-distribution (Fisher distribution),
    which depends on the degrees of freedom :math:`df_1` and
    :math:`df_2` of the two populations. The test statistic is the
    ratio of the two sample variances:

    $$
        F = \\frac{s_1^2}{s_2^2}
    $$

    The two-sided p-value is computed as:

    $$
        p = 2 \\cdot \\min\\bigl(F_{\\text{cdf}}(F),\\; 1 - F_{\\text{cdf}}(F)\\bigr)
    $$

    where $F_{\\text{cdf}}$ is the cumulative distribution function of
    the F-distribution with $df_1 = n_1 - 1$ and $df_2 = n_2 - 1$
    degrees of freedom.

    The null hypothesis $H_0: \\sigma_1^2 = \\sigma_2^2$ is rejected when
    $F < F_{1-\\alpha/2}$ or $F > F_{\\alpha/2}$.

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
        The two-sided p-value for the test.
    F : float
        The F-test statistic.

    Notes
    -----
    The F-test assumes that both samples are drawn from normally
    distributed populations. For non-normal data or heavy-tailed
    distributions, consider using `levene_test` instead.
    """
    F = float(np.var(sample1, ddof=1) / np.var(sample2, ddof=1))
    dof1, dof2 = len(sample1)-1, len(sample2)-1
    cumulated = float(f.cdf(F, dof1, dof2))
    p = 2 * min(cumulated, 1-cumulated)
    return p, F

def t_test(
        sample: NumericSample1D,
        mu: float = 0,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
        ) -> tuple[float, float, int]:
    """Perform one sample t-test.
    
    The t-test tests the null hypothesis that the mean of a sample is
    equal to a given population mean.
    
    Parameters
    ----------
    sample : NumericSample1D
        A one-dimensional array-like object containing the sample.
    mu : float, optional
        The hypothesized mean of the population, by default 0
    alternative : {'two-sided', 'less', 'greater'}, optional
        The alternative hypothesis to use for the test.
        'two-sided' : the mean of the population is not equal to mu
        'less' : the mean of the population is less than mu
        'greater' : the mean of the population is greater than mu
    
    Returns
    -------
    p : float
        The p-value for the test
    t : float
        The t-test statistic
    df : int
        The degrees of freedom
    """
    result = ttest_1samp(sample, mu, alternative=alternative)
    p, t, df = float(result.pvalue), float(result.statistic), int(result.df)  # type: ignore
    return p, t, df

def levene_test(
        sample1: NumericSample1D,
        sample2: NumericSample1D,
        heavy_tailed: bool = False
        ) -> tuple[float, float]:
    """Levene test for equal variances (variance homogeneity).

    The Levene test checks the null hypothesis that all input samples
    are drawn from populations with equal variances. It is a robust
    alternative to the F-test and does not require normality.

    Given a variable *Y* with *N* observations divided into *k* groups
    (where $N_i$ is the size of the *i*-th group), the Levene
    statistic *W* is defined as:

    $$
        W = \\frac{N - k}{k - 1} \\cdot
            \\frac{\\sum_{i=1}^{k} N_i (\\bar{Z}_i - \\bar{Z})^2}
                  {\\sum_{i=1}^{k} \\sum_{j=1}^{N_i} (Z_{ij} - \\bar{Z}_i)^2}
    $$

    where $\\bar{Z}_i$ is the group mean of the $Z_{ij}$ values and
    $\\bar{Z}$ is the overall mean. The $Z_{ij}$ values are absolute
    deviations from a group centre measure:

    $$
        Z_{ij} = \\lvert Y_{ij} - \\tilde{Y}_i \\rvert
    $$

    The choice of centre measure controls the robustness of the test:

    - **Median** ($\\tilde{Y}_i$ = group median) — recommended for
      skewed distributions (default when ``heavy_tailed=False``).
    - **Trimmed mean** (10 % trimmed, $\\tilde{Y}_i'$) — recommended
      for heavy-tailed (leptokurtic) distributions (used when
      ``heavy_tailed=True``).
    - **Mean** ($\\bar{Y}_i$) — original Levene definition, best for
      symmetric, non-heavy-tailed data.

    Parameters
    ----------
    sample1 : NumericSample1D
        A one-dimensional array-like object containing the first sample.
    sample2 : NumericSample1D
        A one-dimensional array-like object containing the second
        sample.
    heavy_tailed : bool, optional
        If ``True``, a 10 % trimmed mean is used as the centre measure
        (robust against heavy-tailed distributions). If ``False``
        (default), the median is used.

    Returns
    -------
    p : float
        p-value for the test.
    L : float
        Levene test statistic *W*.
    """
    center = 'trimmed' if heavy_tailed else 'median'
    L, p = levene(sample1, sample2, center=center)
    return p, L

def variance_stability_test(
        sample: NumericSample1D,
        n_sections: int = 3
        ) -> tuple[float, float]:
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
        ) -> tuple[float, float]:
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
        ) -> tuple[float, float, str]:
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
        ) -> tuple[float, float, str]:
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
        ) -> tuple[float, float, str]:
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
        statistic, p = test_proportions_2indep(
            events1, observations1, events2, observations2, 
            method='wald', alternative='two-sided', return_results=False)
        test = 'Wald'
    return p, statistic, test # type: ignore

def kurtosis_test(
        sample: NumericSample1D
        ) -> tuple[float, float]:
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
        ) -> tuple[float, float]:
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


def dunn_test(
        groups: dict[str, NumericSample1D],
        p_adjust: Literal['bonferroni', 'holm', 'hochberg', 'hommel', 
                          'BH', 'BY', 'none'] = 'bonferroni'
        ) -> pd.DataFrame:
    """Dunn's test for pairwise comparisons after Kruskal-Wallis.
    
    Dunn's test is a post-hoc test used after a significant 
    Kruskal-Wallis test to determine which groups differ from each 
    other. It performs pairwise comparisons using rank sums with 
    multiple comparison correction.
    
    This test is appropriate for:
    - Non-parametric data (no normality assumption)
    - Unbalanced designs (different sample sizes per group)
    - Ordinal or continuous data
    - Multiple group comparisons
    
    Parameters
    ----------
    groups : dict[str, NumericSample1D]
        Dictionary mapping group names to sample data.
        Example: {'Group1': [1, 2, 3], 'Group2': [4, 5, 6]}
    p_adjust : str, optional
        Method for p-value adjustment. Options:
        - 'bonferroni': Conservative, controls FWER
        - 'holm': Less conservative than Bonferroni
        - 'hochberg': Similar to Holm
        - 'BH' (Benjamini-Hochberg): Controls FDR
        - 'BY' (Benjamini-Yekutieli): More conservative FDR
        - 'none': No adjustment
        Default is 'bonferroni'.
    
    Returns
    -------
    DataFrame
        Results table with columns:
        - 'Group1': First group in comparison
        - 'Group2': Second group in comparison
        - 'z_statistic': Standardized test statistic
        - 'p_raw': Unadjusted p-value
        - 'p_adjusted': Adjusted p-value
        - 'significant': Boolean, True if p_adjusted < 0.05
    
    Examples
    --------
    After a significant Kruskal-Wallis test:
    
    ```python
    import daspi as dsp
    import pandas as pd
    
    # Example data: defect counts for 3 factor combinations
    data = {
        'A1_B1': [12, 10, 11, 13],
        'A1_B2': [8, 7, 9, 8],
        'A2_B1': [5, 4, 6, 5],
    }
    
    # Perform Dunn's test
    results = dsp.dunn_test(data, p_adjust='bonferroni')
    print(results)
    ```
    
    Notes
    -----
    Dunn's test compares rank sums between groups using the formula:
    
    $$
    z_{ij} = \\frac{\\bar{R}_i - \\bar{R}_j}{SE}
    $$
    
    where $\\bar{R}_i$ and $\\bar{R}_j$ are mean ranks for groups i and j,
    and SE is the standard error accounting for ties.
    
    **Multiple comparison correction:**
    - Bonferroni: $p_{adj} = \\min(p_{raw} \\times k, 1)$ where k = number of comparisons
    - Benjamini-Hochberg: Controls False Discovery Rate (FDR)
    
    **When to use which correction:**
    - Bonferroni/Holm: When you want strong control of Type I error (FWER)
    - BH/BY: When you're willing to accept some false positives (FDR)
    
    References
    ----------
    Dunn, O. J. (1964). Multiple comparisons using rank sums.
    Technometrics, 6(3), 241-252.
    """
    from scipy.stats import rankdata
    from statsmodels.stats.multitest import multipletests
    
    # Prepare data
    group_names = list(groups.keys())
    n_groups = len(group_names)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for pairwise comparisons")
    
    # Combine all data and compute ranks
    all_data = []
    group_sizes = {}
    group_ranks = {}
    
    for name, data in groups.items():
        data_array = np.asarray(data)
        group_sizes[name] = len(data_array)
        all_data.extend(data_array)
    
    # Compute overall ranks (handling ties)
    all_ranks = rankdata(all_data)
    
    # Assign ranks back to groups
    start_idx = 0
    for name, size in group_sizes.items():
        group_ranks[name] = all_ranks[start_idx:start_idx + size]
        start_idx += size
    
    # Total sample size
    N = len(all_data)
    
    # Count ties for tie correction
    _unique_ranks, counts = np.unique(all_ranks, return_counts=True)
    tie_correction = np.sum(counts ** 3 - counts)
    
    # Compute pairwise comparisons
    results = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            name_i = group_names[i]
            name_j = group_names[j]
            
            # Mean ranks
            R_i = np.mean(group_ranks[name_i])
            R_j = np.mean(group_ranks[name_j])
            
            # Sample sizes
            n_i = group_sizes[name_i]
            n_j = group_sizes[name_j]
            
            # Standard error with tie correction
            SE = np.sqrt(
                (N * (N + 1) / 12 - tie_correction / (12 * (N - 1))) *
                (1 / n_i + 1 / n_j)
            )
            
            # Z-statistic
            z_stat = (R_i - R_j) / SE
            
            # Two-tailed p-value
            p_raw = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results.append({
                'Group1': name_i,
                'Group2': name_j,
                'z_statistic': z_stat,
                'p_raw': p_raw
            })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Adjust p-values
    if p_adjust != 'none':
        # Map user-friendly names to statsmodels names
        method_map = {
            'BH': 'fdr_bh',
            'BY': 'fdr_by',
            'bonferroni': 'bonferroni',
            'holm': 'holm',
            'hochberg': 'simes-hochberg',
            'hommel': 'hommel'
        }
        sm_method = method_map.get(p_adjust, p_adjust)
        
        _, p_adjusted, _, _ = multipletests(
            df_results['p_raw'],
            method=sm_method
        )
        df_results['p_adjusted'] = p_adjusted
    else:
        df_results['p_adjusted'] = df_results['p_raw']
    
    # Mark significant results
    df_results['significant'] = df_results['p_adjusted'] < 0.05
    
    return df_results


def pairwise_tests(
        groups: dict[str, NumericSample1D],
        test: Literal['t', 'mannwhitneyu', 'auto'] = 'auto',
        p_adjust: Literal['bonferroni', 'holm', 'hochberg', 'hommel',
                          'BH', 'BY', 'none'] = 'bonferroni',
        equal_var: bool = True
        ) -> pd.DataFrame:
    """Perform pairwise comparisons between groups with multiple test correction.
    
    This function performs all pairwise comparisons between groups and 
    applies multiple comparison correction. It can automatically select 
    between parametric (t-test) and non-parametric (Mann-Whitney U) tests 
    based on normality.
    
    Parameters
    ----------
    groups : dict[str, NumericSample1D]
        Dictionary mapping group names to sample data.
    test : Literal['t', 'mannwhitneyu', 'auto'], optional
        Statistical test to use:
        - 't': Independent samples t-test (parametric)
        - 'mannwhitneyu': Mann-Whitney U test (non-parametric)
        - 'auto': Automatically select based on normality (default)
    p_adjust : str, optional
        Method for p-value adjustment (same options as dunn_test).
        Default is 'bonferroni'.
    equal_var : bool, optional
        For t-test only: assume equal variances. Default is True.
    
    Returns
    -------
    DataFrame
        Results table with columns similar to dunn_test plus:
        - 'test_used': Which test was applied
        - 'mean_diff': Difference in means (for t-test)
    
    Examples
    --------
    ```python
    import daspi as dsp
    
    data = {
        'Control': [10, 12, 11, 13, 12],
        'Treatment1': [15, 16, 14, 17, 15],
        'Treatment2': [8, 9, 7, 10, 8]
    }
    
    # Automatic test selection with Bonferroni correction
    results = dsp.pairwise_tests(data, test='auto', p_adjust='bonferroni')
    print(results)
    
    # Force Mann-Whitney U test (non-parametric)
    results_np = dsp.pairwise_tests(data, test='mannwhitneyu')
    print(results_np)
    ```
    
    Notes
    -----
    **Test selection (when test='auto'):**
    - Checks normality of all groups using Anderson-Darling test
    - If all groups are normal → uses t-test
    - If any group is non-normal → uses Mann-Whitney U
    
    **Multiple comparison methods:**
    - Use Bonferroni for strong Type I error control
    - Use BH (Benjamini-Hochberg) for more power when many comparisons
    
    **Interpretation:**
    - p_adjusted < 0.05: Groups significantly different at α=0.05 level
    - mean_diff: Positive means Group1 > Group2
    """
    from statsmodels.stats.multitest import multipletests
    
    group_names = list(groups.keys())
    n_groups = len(group_names)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for pairwise comparisons")
    
    # Auto-select test based on normality if requested
    if test == 'auto':
        all_normal_flag = all_normal(*groups.values())
        test = 't' if all_normal_flag else 'mannwhitneyu'
    
    # Perform pairwise tests
    results = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            name_i = group_names[i]
            name_j = group_names[j]
            
            sample_i = np.asarray(groups[name_i])
            sample_j = np.asarray(groups[name_j])
            
            if test == 't':
                # Independent samples t-test
                t_stat, p_raw = ttest_ind(
                    sample_i, sample_j, equal_var=equal_var)
                statistic = t_stat
                mean_diff = np.mean(sample_i) - np.mean(sample_j)
                test_used = 't-test'
            
            elif test == 'mannwhitneyu':
                # Mann-Whitney U test
                u_stat, p_raw = mannwhitneyu(
                    sample_i, sample_j, alternative='two-sided')
                statistic = u_stat
                mean_diff = np.median(sample_i) - np.median(sample_j)
                test_used = 'Mann-Whitney U'
            
            else:
                raise ValueError(f"Unknown test: {test}")
            
            results.append({
                'Group1': name_i,
                'Group2': name_j,
                'statistic': statistic,
                'mean_diff': mean_diff,
                'p_raw': p_raw,
                'test_used': test_used
            })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Adjust p-values
    if p_adjust != 'none':
        # Map user-friendly names to statsmodels names
        method_map = {
            'BH': 'fdr_bh',
            'BY': 'fdr_by',
            'bonferroni': 'bonferroni',
            'holm': 'holm',
            'hochberg': 'simes-hochberg',
            'hommel': 'hommel'
        }
        sm_method = method_map.get(p_adjust, p_adjust)
        
        _, p_adjusted, _, _ = multipletests(
            df_results['p_raw'],
            method=sm_method
        )
        df_results['p_adjusted'] = p_adjusted
    else:
        df_results['p_adjusted'] = df_results['p_raw']
    
    # Mark significant results
    df_results['significant'] = df_results['p_adjusted'] < 0.05
    
    return df_results
