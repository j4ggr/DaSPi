# source for ci: https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#conf_int_of_var
import warnings

import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Type
from typing import Self
from typing import Tuple
from typing import Literal
from typing import overload
from typing import Callable
from numpy.typing import NDArray
from numpy.linalg import LinAlgError
from scipy.interpolate import interp1d
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.stats._distn_infrastructure import rv_continuous

from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis

from .._typing import NumericSample1D

from ..constants import DIST
from ..constants import ANOVA
from ..constants import DEFAULT
from ..constants import SIGMA_DIFFERENCE

from .montecarlo import SpecLimits
from .montecarlo import Specification

from .confidence import cp_ci
from .confidence import cpk_ci
from .confidence import mean_ci
from .confidence import stdev_ci
from .confidence import median_ci
from .confidence import confidence_to_alpha

from .hypothesis import t_test
from .hypothesis import skew_test
from .hypothesis import kurtosis_test
from .hypothesis import ensure_generic
from .hypothesis import mean_stability_test
from .hypothesis import anderson_darling_test
from .hypothesis import variance_stability_test
from .hypothesis import kolmogorov_smirnov_test


class DistributionEstimator:
    """A class to estimate the distribution of a given 1D numeric sample.
    
    This class provides methods to estimate distribution by fitting a
    continuous distribution from `scipy.stats` to the provided samples.
    It uses the Kolmogorov-Smirnov test to evaluate the fit of the
    distribution to the data. The distribution with a higher p-value
    is considered a better fit.
    
    Parameters
    ----------
    samples : NumericSample1D
        The 1D numeric sample for which the distribution is to be 
        estimated. This should be a Series or array-like object
        containing numeric values.
    dist : str rv_continuous, optional
        Distributions to which the data may be subject. Only continuous 
        distributions of scipy.stats are allowed. Default is 'norm'
    possible_dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default `DIST.COMMON`
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        How to handle NaN values in the samples. 
        - 'propagate': NaN values are preserved in the analysis.
        - 'raise': Raises an error if NaN values are found.
        - 'omit': Omits NaN values from the analysis, default is 'omit'.
    
    Raises
    ------
    ValueError
        If NaN values are found in the samples and `nan_policy` is set to 
        'raise'.
    UserWarning
        If NaN values are found in the samples and `nan_policy` is set 
        to 'omit' or 'propagate'. The warning indicates that NaN values 
        will be omitted from the analysis or may lead to unexpected 
        results.
    """
    __slots__ = (
        '_samples',
        '_filtered',
        '_n_samples',
        '_n_missing',
        '_dist',
        '_p_dist',
        '_D',
        '_params',
        '_excess',
        '_p_excess', 
        '_skew',
        '_p_skew',
        '_p_ad',
        '_ss',
        '_aic',
        '_bic',
        '_descriptive_statistic_attrs',
        'possible_dists',
        'nan_policy')

    _samples: pd.Series
    _filtered: pd.Series
    _n_samples: int
    _n_missing: int
    _dist: rv_continuous | None
    _p_dist: float | None
    _D: float | None
    _params: Tuple[float, ...] | None
    _excess: float | None
    _p_excess: float | None
    _skew: float | None
    _p_skew: float | None
    _p_ad: float | None
    _ss: float | None
    _aic: float | None
    _bic: float | None
    _descriptive_statistic_attrs: Tuple[str, ...]
    possible_dists: Tuple[str | rv_continuous, ...]
    """Distributions given during initialization to which the data may 
    be subject."""
    nan_policy : Literal['propagate', 'raise', 'omit']
    """How to handle NaN values in the samples. 
        - 'propagate': NaN values are preserved in the analysis.
        - 'raise': Raises an error if NaN values are found.
        - 'omit': Omits NaN values from the analysis, default is 'omit'."""

    def __init__(
            self, 
            samples: NumericSample1D, 
            dist: str | rv_continuous | None = None,
            possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON,
            nan_policy: Literal['propagate', 'raise', 'omit'] = 'omit',
            ) -> None:
        if not isinstance(samples, pd.Series):
            samples = pd.Series(samples)
        has_nan = samples.isna().any()
        if has_nan and nan_policy == 'raise':
            raise ValueError(
                'NaN values found in the samples. '
                'Set nan_policy to "omit" to ignore them.')
        elif nan_policy == 'omit':
            if has_nan:
                warnings.warn(
                    'NaN values found in the samples. '
                    'These will be omitted from the analysis.',
                    UserWarning)
        elif nan_policy == 'propagate' and has_nan:
            warnings.warn(
                'NaN values found in the samples. '
                'This may lead to unexpected results.',
                UserWarning)
        self.nan_policy = nan_policy
        self._filtered = pd.Series()
        if not isinstance(samples, pd.Series):
            samples = pd.Series(samples)
        self._samples = samples
        self._n_samples = len(self.samples)
        self._n_missing = self.samples.isna().sum()
        self._dist = None
        self._p_dist = None
        self._params = None
        self._D = None
        self.possible_dists = possible_dists
        self.dist = dist

        self._excess = None
        self._p_excess = None
        self._skew = None
        self._p_skew = None
        self._p_ad = None

        self._ss = None
        self._aic = None
        self._bic = None

        self._descriptive_statistic_attrs = (
            'n_samples',
            'n_missing',
            'dist',
            'p_dist',
            'p_ad',
            'excess',
            'p_excess',
            'skew',
            'p_skew',)
    
    @property
    def descriptive_statistic_attrs(self) -> Tuple[str, ...]:
        """Get attribute names used for `describe` method."""
        return self._descriptive_statistic_attrs
    
    @property
    def samples(self) -> pd.Series:
        """Get the raw samples as it was given during instantiation
        as pandas Series (read-only)."""
        return self._samples
    
    @property
    def filtered(self) -> pd.Series:
        """Get the samples without error values and no missing value 
        (read-only)."""
        if self._filtered.empty:
            self._filtered = pd.to_numeric(self.samples[self.samples.notna()])
        return self._filtered

    @property
    def n_samples(self) -> int:
        """Get sample size of unfiltered samples (read-only)."""
        return self._n_samples

    @property
    def n_missing(self) -> int:
        """Get amount of missing values (read-only)."""
        return self._n_missing
    
    @property
    def dist(self) -> rv_continuous:
        """This is the generic continuous distribution class of the 
        provided or evaluated distribution."""
        if self._dist is None:
            self._dist, self._p_dist, self._params = self.distribution()
        return self._dist
    @dist.setter
    def dist(self, dist: str | rv_continuous | None) -> None:
        """Set the distribution to be used for estimation. If a string 
        is provided, it will be converted to a continuous distribution 
        class using `ensure_generic`. If None, the distribution will be 
        estimated from the samples."""
        if isinstance(dist, str):
            dist = ensure_generic(dist)
        self._dist = dist
        if self._dist != dist:
            self._p_dist = None
            self._params = None
            self._D = None
    
    @property
    def p_dist(self) -> float:
        """Get the two-tailed p-value for the provided or fitted 
        distribution. A higher p-value indicates a better fit to the 
        data (read-only)."""
        if self._p_dist is None:
            self._dist, self._p_dist, self._params = self.distribution()
        return self._p_dist
    
    @property
    def D(self) -> float:
        """Get the Kolmogorov-Smirnov test statistic, either D, D+ or 
        D-."""
        if self._D is None:
            self._D = kolmogorov_smirnov_test(self.filtered, self.dist)[0]
        return self._D

    @property
    def params(self) -> Tuple[float, ...]:
        """Estimates for any distribution shape parameters (if 
        applicable), followed by those for location and scale. For most 
        random variables, shape statistics will be returned, but there 
        are exceptions (e.g. norm). Can be used to generate values with 
        the help of the dist attribute (read-only)."""
        if self._params is None:
            self._dist, self._p_dist, self._params = self.distribution()
        return self._params

    @property
    def excess(self) -> float:
        """Get the Fisher kurtosis (excess) of filtered samples.
        Calculations are corrected for statistical bias (read-only).
        The curvature of the distribution corresponds to the 
        curvature of a normal distribution when the excess is close to 
        zero. Distributions with negative excess kurtosis are said to be 
        platykurtic, this distribution produces fewer and/or less 
        extreme outliers than the normal distribution (e.g. the uniform 
        distribution has no outliers). Distributions with a positive 
        excess kurtosis are said to be leptokurtic (e.g. the Laplace 
        distribution, which has tails that asymptotically approach zero 
        more slowly than a Gaussian, and therefore produces more 
        outliers than the normal distribution):
            - excess < 0: less extreme outliers than normal distribution
            - excess > 0: more extreme outliers than normal distribution
        """
        if self._excess is None:
            self._excess = float(kurtosis(
                self.filtered, fisher=True, bias=False))
        return self._excess

    @property
    def p_excess(self) -> float:
        """Get the probability that the excess of the population that
        the sample was drawn from is the same as that of a corresponding
        normal distribution (read-only)."""
        if self._p_excess is None:
            self._p_excess = kurtosis_test(self.filtered)[0]
        return self._p_excess

    @property
    def skew(self) -> float:
        """Get the skewness of the filtered samples (read-only).
        Calculations are corrected for statistical bias.
        For normally distributed data, the skewness should be about zero. 
        For unimodal continuous distributions, a skewness value greater 
        than zero means that there is more weight in the right tail of 
        the distribution:
            - skew < 0: left-skewed -> long tail left
            - skew > 0: right-skewed -> long tail right
        """
        if self._skew is None:
            self._skew = float(skew(self.filtered, bias=False))
        return self._skew

    @property
    def p_skew(self) -> float:
        """Get the probability that the skewness of the population that
        the sample was drawn from is the same as that of a corresponding
        normal distribution (read-only)."""
        if self._p_skew is None:
            self._p_skew = skew_test(self.filtered)[0]
        return self._p_skew

    @property
    def p_ad(self) -> float:
        """Get the probability that the filtered samples are subject of
        the normal distribution by performing a Anderson-Darling test
        (read-only)."""
        if self._p_ad is None:
            self._p_ad = anderson_darling_test(self.filtered)[0]
        return self._p_ad
    
    def mask_missing(self) -> 'Series[bool]':
        """Returns a boolean mask indicating which samples are missing 
        (NaN).

        This mask can be used to filter or analyze entries in the 
        `samples` attribute that contain missing values.

        Returns
        -------
        Series[bool]:
            A boolean Series where True indicates a missing sample.
        """
        return self.samples.isna()
    
    def mask_ok(self) -> Series:
        """Returns a boolean mask indicating which samples are valid 
        (OK).

        A sample is considered OK if it is:
        - Not missing (i.e., not NaN)

        Returns
        -------
        Series[bool]
            A boolean Series where True indicates a valid sample.
        """
        return ~self.mask_missing()
    
    def distribution(self) -> Tuple[rv_continuous, float, Tuple[float, ...]]:
        """Estimate the distribution by selecting the one from the
        provided distributions that best reflects the filtered data.

        Returns
        -------
        dist : scipy.stats rv_continuous
            A generic continous distribution class of best fit
        p : float
            The two-tailed p-value for the best fit
        params : Tuple[float, ...]
            Estimates for any shape parameters (if applicable), followed
            by those for location and scale. For most random variables,
            shape statistics will be returned, but there are exceptions
            (e.g. norm). Can be used to generate values with the help of
            returned dist
        
        Notes
        -----
        First, the p-score is calculated by performing a 
        Kolmogorov-Smirnov test to determine how well each distribution 
        fits the samples. Whatever has the highest P-score is considered
        the most accurate. This is because a higher p-score means the 
        hypothesis is closest to reality."""
        samples = self.filtered if self.nan_policy == 'omit' else self.samples
        dists =  self.possible_dists if self._dist is None else (self._dist, )
        return estimate_distribution(samples, dists)
    
    def stable_variance(
            self, alpha: float = 0.05, n_sections : int = 3) -> bool:
        """Test whether the variance remains stable across the samples. 
        
        The sample data is divided into subgroups and the variances of
        their sections are checked using the Levene test.
        
        Parameters
        ----------
        alpha : float
            Alpha risk of hypothesis tests. If a p-value is below this 
            limit, the null hypothesis is rejected
        n_sections : int, optional
            Amount of sections to divide the filtered samples into, 
            by default 3
        
        Returns
        -------
        stable : bool
            True if the p-value > alpha
        """
        assert isinstance(n_sections, int)
        assert  1 < n_sections < len(self.filtered)
        p, _ = variance_stability_test(self.filtered, n_sections=n_sections)
        return p > alpha
    
    def stable_mean(
            self, alpha: float = 0.05, n_sections : int = 3) -> bool:
        """Test whether the mean remains stable across the samples. 
        
        The sample data is divided into subgroups and the mean of their 
        sections are checked using the F test.
        
        Parameters
        ----------
        alpha : float
            Alpha risk of hypothesis tests. If a p-value is below this 
            limit, the null hypothesis is rejected
        n_sections : int, optional
            Amount of sections to divide the filtered samples into, 
            by default 3
        
        Returns
        -------
        stable : bool
            True if the p-value > alpha
        """
        assert isinstance(n_sections, int)
        assert  1 < n_sections < len(self.filtered)
        p, _ = mean_stability_test(self.filtered, n_sections=n_sections)
        return p > alpha
    
    def follows_norm_curve(
            self, alpha: float = 0.05, excess_test: bool = True,
            skew_test: bool = True, ad_test: bool = False) -> bool:
        """Checks whether the sample data is subject to normal 
        distribution by performing one or more of the following tests 
        (depending on the input):
        - Skewness test
        - Bulge test
        - Anderson-Darling test
        
        Parameters
        ----------
        alpha : float
            Alpha risk of hypothesis tests. If a p-value is below this 
            limit, the null hypothesis is rejected
        skew_test : bool, optional
            If true, an skew test will also be carried out, by default 
            True
        ad_test : bool, optional
            If true, an excess test will also be carried out, by default True
        ad_test : bool, optional
            If true, an Anderson Darling test will also be carried out,
            by default False
            
        Returns
        -------
        remain_h0 : bool
            True if all p-values of the tests performed are greater than 
            alpha, otherwise False
        """
        remain_h0 = [
            (self.p_excess > alpha) if excess_test else True,
            (self.p_skew > alpha) if skew_test else True,
            (self.p_ad > alpha) if ad_test else True]
        return all(remain_h0)
    
    def _get_descriptive_attr_(self, name) -> float | int | str:
        """Return the current value of the specified attribute."""
        assert name in self.descriptive_statistic_attrs, (
            f'Attribute {name} is not a valid descriptive statistic '
            'attribute')
        if name == 'dist':
            return self.dist.name
        return getattr(self, name)
    
    def describe(self, exclude: Tuple[str, ...] = ()) -> DataFrame:
        """Generate descriptive statistics.
        
        Parameters
        ----------
        exclude : Tuple[str,...], optional
            Attributes to exclude from the summary statistics,
            by default ()
        
        Returns
        -------
        stats : DataFrame
            Summary statistics as pandas DataFrame. The indices of the
            DataFrame are the attributes that have been computed and the
            column name is the name of the samples.
        """
        names = (
            n for n in self.descriptive_statistic_attrs if n not in exclude)
        data = pd.DataFrame(
            data={name: [self._get_descriptive_attr_(name)] for name in names},
            index=[self.samples.name])
        return data.T


class Estimator(DistributionEstimator):
    """An object for various statistical estimators
    
    The attributes are calculated lazily. After the class is 
    instantiated, all attributes are set to None. As soon as an 
    attribute (actually Property) is called, the value is calculated
    and stored so that the calculation is only performed once
    
    Parameters
    ----------
    samples : NumericSample1D
        sample data
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the control 
        limits (process spread):
        - `eval`: The strategy is determined according to the given 
          evaluate function. If none is given, the internal `evaluate`
          method is used.
        - `fit`: First, the distribution that best represents the 
          process data is searched for and then the agreed process 
          spread is calculated
        - `norm`: it is assumed that the data is subject to normal 
          distribution. The variation tolerance is then calculated as 
          agreement * standard deviation
        - `data`: The quantiles for the process variation tolerance 
          are read directly from the data.
        
        Default is 'norm'.
    agreement : int or float, optional
        Specify the tolerated process variation for which the 
        control limits are to be calculated. 
        - If int, the spread is determined using the normal 
          distribution agreement*σ, 
          e.g. agreement = 6 -> 6*σ ~ covers 99.75 % of the data. 
          The upper and lower permissible quantiles are then 
          calculated from this.
        - If float, the value must be between 0 and 1.This value is
          then interpreted as the acceptable proportion for the 
          spread, e.g. 0.9973 (which corresponds to ~ 6 σ)
        
        Default is 6 because SixSigma ;-)
    possible_dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default `DIST.COMMON`
    
    Examples
    --------

    ```python
    import numpy as np
    import daspi as dsp

    np.random.seed(1)
    samples = data = np.random.weibull(a=1.5, size=100)
    estimation = dsp.Estimator(
        samples=samples,
        strategy='fit',
        agreement=6,
        possible_dists=dsp.DIST.COMMON_NOT_NORM)
    print(estimation.describe())
    ```

    So you will receive the following output:

    ```console
                      None
    n_samples          100
    n_missing            0
    min           0.002356
    max           2.724595
    R             2.722239
    mean           0.86943
    median         0.74006
    std           0.593666
    sem           0.059367
    excess        0.163836
    p_excess      0.599041
    skew          0.802202
    p_skew        0.001918
    p_ad          0.000455
    dist       weibull_min
    p_dist        0.968613
    strategy           fit
    lcl          -0.004917
    ucl            3.43952
    ```

    Notes
    -----
    A special case occurs when the agreement is 1. For a corresponding
    standard deviation, enter 1 as an integer. If you want percentiles 
    or the entire range, enter it as a floating-point number (1.0) or as 
    `float('inf')`. If strategy is 'data', `lcl` and `ucl` correspond to 
    `min` and `max`, otherwise we get `-inf` and `inf`.
    """
    __slots__ = (
        '_min',
        '_max', 
        '_R',
        '_mean',
        '_median',
        '_std',
        '_sem',
        '_lcl',
        '_ucl',
        '_strategy',
        '_agreement',
        '_k',
        '_evaluate',
        '_q_low',
        '_q_upp')
    
    _min: float | None
    _max: float | None
    _R: float | None
    _mean: float | None
    _median: float | None
    _std: float | None
    _sem: float | None
    _lcl: float | None
    _ucl: float | None
    _strategy: Literal['eval', 'fit', 'norm', 'data'] 
    _agreement: int | float
    _k: float
    _evaluate: Callable | None
    _q_low: float | None
    _q_upp: float | None

    def __init__(
            self,
            samples: NumericSample1D, 
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: int | float = 6, 
            possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON,
            evaluate: Callable | None = None
            ) -> None:
        super().__init__(
            samples=samples,
            dist=None,
            possible_dists=possible_dists,
            nan_policy='omit')
        self._min = None
        self._max = None
        self._R = None
        self._mean = None
        self._median = None
        self._std = None
        self._sem = None
        self._lcl = None
        self._ucl = None
        self._q_low = None
        self._q_upp = None
        self._strategy = 'norm'
        self.strategy = strategy
        self._agreement = -1
        self.agreement = agreement
        self._evaluate = evaluate
        self._descriptive_statistic_attrs = (
            'n_samples',
            'n_missing',
            'min',
            'max',
            'R',
            'mean',
            'median',
            'std',
            'sem',
            'excess',
            'p_excess',
            'skew',
            'p_skew',
            'p_ad',
            'dist',
            'p_dist',
            'strategy',
            'lcl',
            'ucl')

    @property
    def dof(self) -> int:
        """Get degree of freedom for filtered samples (read-only)."""
        return len(self.filtered)-1

    @property
    def min(self) -> float:
        """Get the minimum value of filtered samples (read-only)."""
        if self._min is None:
            self._min = float(self.filtered.min())
        return self._min

    @property
    def max(self) -> float:
        """Get the maximum value of filtered samples (read-only)."""
        if self._max is None:
            self._max = float(self.filtered.max())
        return self._max
    
    @property
    def R(self) -> float:
        """Get range of filtered samples (read-only)."""
        if self._R is None:
            self._R = self.max - self.min
        return self._R

    @property
    def mean(self) -> float:
        """Get mean of filtered samples (read-only)."""
        if self._mean is None:
            self._mean = self.filtered.mean()
        return self._mean

    @property
    def median(self) -> float:
        """Get median of filtered samples (read-only)."""
        if self._median is None:
            self._median = self.filtered.median()
        return self._median

    @property
    def std(self) -> float:
        """Get standard deviation of filtered samples (read-only)."""
        if self._std is None:
            self._std = self.filtered.std()
        return self._std

    @property
    def sem(self) -> float:
        """Get standard error mean of filtered samples (read-only)."""
        if self._sem is None:
            self._sem = float(self.filtered.sem()) # type: ignore
        return self._sem

    @property
    def lcl(self) -> float:
        """Get lower control limit according to given strategy and 
        agreement (read-only)."""
        if self._lcl is None:
            self._lcl, self._ucl = self._calculate_control_limits_()
        return self._lcl

    @property
    def ucl(self) -> float:
        """Get upper control limit according to given strategy and 
        agreement (read-only)."""
        if self._ucl is None:
            self._lcl, self._ucl = self._calculate_control_limits_()
        return self._ucl

    @property
    def q_low(self) -> float:
        """Get quantil for lower control limit according to given 
        agreement. If the samples is subject to normal distribution and 
        the agreement is given as 6, this value corresponds to the 
        0.135 % quantile: 6 σ ~ 99.73 % of the samples (read-only)."""
        if self._q_low is None:
            self._q_low = float(stats.norm.cdf(-self.agreement / 2))
        return self._q_low

    @property
    def q_upp(self) -> float:
        """Get quantil for upper control limit according to given 
        agreement. If the sample data is subject to normal distribution 
        and the agreement is given as 6, this value corresponds to the 
        Q_0.99865: 0.99865-quantile or 99.865-percentile (read-only)."""
        if self._q_upp is None:
            self._q_upp = 1 - self.q_low
        return self._q_upp

    @property
    def strategy(self) -> Literal['eval', 'fit', 'norm', 'data']:
        """Strategy used to determine the control limits. The control 
        limits can also be interpreted as the process range.

        Set strategy as one of {'eval', 'fit', 'norm', 'data'}
            - eval: If no evaluate function is given, the strategy is 
            determined according to the internal evaluate method. 
            - fit: First, the distribution is searched for that best 
            represents the process data and then the process variation 
            tolerance is calculated
            - norm: it is assumed that the data is subject to normal 
            distribution. The variation tolerance is then calculated as 
            agreement * standard deviation
            - data: The quantiles for the process variation tolerance 
            are read directly from the samples."""
        return self._strategy
    @strategy.setter
    def strategy(self, strategy: Literal['eval', 'fit', 'norm', 'data']) -> None:
        assert strategy in ['eval', 'fit', 'norm', 'data']
        if self._strategy != strategy:
            self._strategy = strategy
            self._reset_values_()

    @property
    def agreement(self) -> float:
        """Get the multiplier of the σ agreement for Cp and Cpk 
        value (default 6). By setting this value the cp and cpk values
        are resetted to None.
        
        If setting agreement by giving the percentile, enter the 
        acceptable proportion for the spread, e.g. 0.9973 
        (which corresponds to ~ 6 σ).
        
        A special case occurs when the agreement is 1. For a
        corresponding standard deviation, enter 1 as an integer. If you
        want percentiles or the entire range, enter it as a
        floating-point number (1.0) or as `float('inf')`."""
        return self._agreement
    @agreement.setter
    def agreement(self, agreement: int | float) -> None:
        assert agreement > 0, (
            'Agreement must be set as a percentage (0.0 < agreement <= 1.0) '
            + 'or as a multiple of the standard deviation (agreement >= 1), '
            + f'got {agreement}.')
        
        is_percentile = (
            agreement < 1 or (agreement == 1 and isinstance(agreement, float)))
        if is_percentile:
            self._k = float(stats.norm.ppf((1 + agreement) / 2))
            agreement = 2 * self._k
        else:
            self._k = agreement / 2
        
        if self._agreement != agreement:
            self._agreement = agreement
            self._reset_values_()

    def z_transform(self, x: float) -> float:
        """Transform value to z-score.
        
        This method produces a value from a distribution with a mean of 
        0 and a standard deviation of 1. The value indicates how many
        standard deviations the value is from the mean.

        Parameters
        ----------
        x : float
            value to be transformed

        Returns
        -------
        z : float
            z-score
        """
        return (x - self.mean) / self.std

    def mean_ci(self, level: float = 0.95) -> Tuple[float, float]:
        """Two sided confidence interval for mean of filtered data

        Parameters
        ----------
        level : float in (0, 1), optional
            confidence level, by default 0.95

        Returns
        -------
        ci_low, ci_upp : float
            lower and upper confidence level
        """
        ci_low, ci_upp = mean_ci(sample=self.filtered, level=level)[1:]
        return ci_low, ci_upp

    def median_ci(self, level: float = 0.95) -> Tuple[float, float]:
        """Two sided confidence interval for median of filtered data

        Parameters
        ----------
        level : float in (0, 1), optional
            confidence level, by default 0.95

        Returns
        -------
        ci_low, ci_upp : float
            lower and upper confidence level
        """
        ci_low, ci_upp = median_ci(sample=self.filtered, level=level)[1:]
        return ci_low, ci_upp

    def stdev_ci(self, level: float = 0.95) -> Tuple[float, float]:
        """Two sided confidence interval for standard deviation of 
        filtered data

        Parameters
        ----------
        level : float in (0, 1), optional
            confidence level, by default 0.95

        Returns
        -------
        ci_low, ci_upp : float
            lower and upper confidence level
        """
        ci_low, ci_upp = stdev_ci(sample=self.filtered, level=level)[1:]
        return ci_low, ci_upp
    
    def evaluate(self) -> Literal['fit', 'norm', 'data']:
        """Evaluate strategy to calculate control limits. If no evaluate
        function is given the strategy is evaluated as follows:
        
        1. If variance is not stable within the samples
        -> strategy = 'data'
        2. If variance and mean is stable and samples follow a normal 
        curve -> strategy = 'norm'
        3. If variance and mean is stable but samples don't follow a
        normal curve -> strategy = 'fit'
        4. If variance is stable but mean not and samples follow a 
        normal curve -> strategy = 'norm'
        5. If variance is stable but mean not and samples don't follow a 
        normal curve -> strategy = 'data'

        Returns
        -------
        strategy : {'fit', 'norm', 'data'}  
            Evaluated strategy to calculate control limits
        """
        if self._evaluate is not None:
            strategy = self._evaluate(self)
            assert strategy in ('fit', 'norm', 'data'), (
                f'Given evaluate function returned {strategy}, '
                '"fit", "norm" or "data" is required')
            return strategy
        
        if not self.stable_variance():
            strategy = 'data'
        elif self.stable_mean():
            strategy = 'norm' if self.follows_norm_curve() else 'fit'
        else:
            strategy = 'norm' if self.follows_norm_curve() else 'data'
        return strategy
    
    def _calculate_control_limits_(self) -> Tuple[float, float]:
        """Calculate the control limits according to given strategy
        
        - `eval`: first evaluate the strategy according to evaluate 
            method. Then the limits are calculated using one of the 
            following points.
        - `fit`: first fit samples to a distribution and then calculate
            quantile for this distribution according to given agreement.
        - `norm`: The control limits are calculated from the mean plus 
            and minus the standard deviation multiplied by the expansion 
            factor k.
        - `data`: The quantiles are calculated directly from the sample
            according the given agreement

        Returns
        -------
        lcl, ucl : float
            control limits also known as process range.
        """
        match self.strategy:
            case 'eval':
                self.strategy = self.evaluate()
                lcl, ucl = self._calculate_control_limits_()
            case 'fit':
                lcl = float(self.dist.ppf(self.q_low, *self.params))
                ucl = float(self.dist.ppf(self.q_upp, *self.params))
            case 'norm':
                lcl = self.mean - self._k * self.std
                ucl = self.mean + self._k * self.std
            case 'data':
                lcl = float(np.quantile(self.filtered, self.q_low))
                ucl = float(np.quantile(self.filtered, self.q_upp))
        return lcl, ucl

    def _reset_values_(self) -> None:
        """Set the control limits and quantiles to None. This function 
        is called when one of the values relevant to the calculation of
        those limits is adjusted (strategy or agreement for the control
        limits)."""
        self._lcl = None
        self._ucl = None
        self._q_low = None
        self._q_upp = None


class ProcessEstimator(Estimator):
    """An object for various statistical estimators. This class extends 
    the estimator with process-specific statistics such as specification
    limits, Cp and Cpk values.
    
    The attributes are calculated lazily. After the class is instantiated,
    all attributes are set to None. As soon as an attribute (actually
    Property) is called, the value is calculated and stored so that the
    calculation is only performed once.
    
    Parameters
    ----------
    samples : NumericSample1D
        1D array of process data.
    spec_limits : SpecLimits
        Specification limits for process data.
    error_values : tuple of float, optional
        If the process data may contain coded values for measurement 
        errors or similar, they can be specified here, 
        by default [].
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the control 
        limits (process spread):
        - `eval`: The strategy is determined according to the given 
        evaluate function. If none is given, the internal `evaluate`
        method is used.
        - `fit`: First, the distribution that best represents the 
        process data is searched for and then the agreed process 
        spread is calculated
        - norm: it is assumed that the data is subject to normal 
        distribution. The variation tolerance is then calculated as 
        agreement * standard deviation
        - data: The quantiles for the process variation tolerance 
        are read directly from the data.
        by default 'norm'
    agreement : float or int, optional
        Specify the tolerated process variation for which the 
        control limits are to be calculated. 
        - If int, the spread is determined using the normal 
        distribution agreement*σ, 
        e.g. agreement = 6 -> 6*σ ~ covers 99.75 % of the data. 
        The upper and lower permissible quantiles are then 
        calculated from this.
        - If float, the value must be between 0 and 1.This value is
        then interpreted as the acceptable proportion for the 
        spread, e.g. 0.9973 (which corresponds to ~ 6 σ)
        by default 6
    possible_dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default DIST.COMMON

    Examples
    --------
    You can get a comprehensive analysis of your process using the 
    `describe()` method, which returns a `pandas.DataFrame`. This 
    contains all important metrics, such as the Cp (if possible) and 
    Cpk values:
    
    ```python
    import daspi as dsp

    df = dsp.load_dataset('drop_card')
    spec_limits = dsp.SpecLimits(0, float(df.loc[0, 'usl']))
    target = 'distance'

    drop_process = dsp.ProcessEstimator(
        samples=df[target],
        spec_limits=spec_limits)
    print(drop_process.describe())
    ```

    However, in this dataset, the cards were dropped in two different 
    ways. To compare both methods, a DataFrame containing both process 
    analyses can be created as follows:

    ```python
    method_mapping = {0: 'parallel', 1: 'perpendicular'}
    samples_parallel = df[df['method']==method_mapping[0]][target]
    samples_series = df[df['method']==method_mapping[1]][target]

    drop_analysis = pd.concat([
        dsp.ProcessEstimator(samples_parallel, spec_limits).describe(),
        dsp.ProcessEstimator(samples_series, spec_limits).describe()],
        axis=1,
        ignore_index=True,
    ).rename(
        columns=method_mapping
    )
    print(drop_analysis)
    ```

    You can get a detailed visual analysis with the precast chart 
    `daspi.plotlib.precast.ProcessCapabilityAnalysisCharts`
    """
    __slots__ = (
        '_spec_limits',
        '_n_ok',
        '_n_nok',
        '_ok',
        '_nok',
        '_nok_norm', 
        '_nok_fit',
        '_error_values',
        '_n_errors',
        '_cp',
        '_cpk',
        '_Z', 
        '_Z_lt')
    
    _spec_limits: SpecLimits
    _n_ok: int | None
    _n_nok: int | None
    _ok: float | None
    _nok: float | None
    _nok_norm: float | None
    _nok_fit: float | None
    _error_values: Tuple[float, ...]
    _n_errrors: int
    _cp: float | None
    _cpk: float | None
    _Z: float | None
    _Z_lt: float | None

    def __init__(
            self,
            samples: NumericSample1D, 
            spec_limits: SpecLimits | Specification,
            error_values: Tuple[float, ...] = (),
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: float | int = 6, 
            possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON
            ) -> None:
        self._n_ok = None
        self._n_nok = None
        self._n_errors = None
        self._ok = None
        self._nok = None
        self._nok_norm = None
        self._nok_fit = None
        self._cp = None
        self._cpk = None
        self._Z = None
        self._Z_lt = None
        self._error_values = error_values
        if isinstance(spec_limits, Specification):
            spec_limits = spec_limits.LIMITS
        self._spec_limits = spec_limits
        self._reset_values_()
        super().__init__(samples, strategy, agreement, possible_dists)
    
    @property
    def descriptive_statistic_attrs(self) -> Tuple[str, ...]:
        """Get attribute names used for `describe` method (read-only)."""
        attrs = (
            'n_samples',
            'n_missing',
            'n_ok',
            'n_nok',
            'n_errors',
            'ok',
            'nok', 
            'nok_norm',
            'nok_fit'
            'min',
            'max',
            'R',
            'mean',
            'median',
            'std',
            'sem',
            'excess',
            'p_excess',
            'skew',
            'p_skew',
            'p_ad',
            'dist',
            'p_dist',
            'strategy',
            'lcl',
            'ucl'
            'lsl',
            'usl',
            'cp',
            'cpk',
            'Z',
            'Z_lt')
        return attrs
        
    @property
    def filtered(self) -> Series:
        """Get the data without error values and no missing value
         (read-only)."""
        if self._filtered.empty:
            self._filtered = pd.to_numeric(
                self.samples[
                    ~ (self.mask_error() | self.mask_missing())])
        return self._filtered
    
    @property
    def n_ok(self) -> int:
        """Get amount of OK-values (read-only)."""
        if self._n_ok is None:
            self._n_ok = int(
                self.n_samples
                - self.n_nok
                - self.n_errors
                - self.n_missing)
        return self._n_ok
    
    @property
    def n_nok(self) -> int:
        """Get amount of NOK-values (read-only)."""
        if self._n_nok is None:
            self._n_nok = int(self.mask_nok().sum())
        return self._n_nok
    
    @property
    def ok(self) -> str:
        """Get amount of OK-values as percent (read-only)."""
        if self._ok is None:
            self._ok = self.n_ok/self.n_samples
        return f'{100 * self._ok:.2f} %'
    
    @property
    def nok(self) -> str:
        """Get amount of NOK-values as percent (read-only)."""
        if self._nok is None:
            self._nok = self.n_nok/self.n_samples
        return f'{100 * self._nok:.2f} %'
    
    @property
    def nok_norm(self) -> str:
        """Predict the amount NOK-values as percent based on the 
        norm distribution (read-only)."""
        if self._nok_norm is None:
            self._nok_norm = float(
                1 - norm.cdf(self.usl, loc=self.mean, scale=self.std)
                + norm.cdf(self.lsl, loc=self.mean, scale=self.std))
        return f'{100 * self._nok_norm:.2f} %'
    
    @property
    def nok_fit(self) -> str:
        """Predict the amount NOK-values as percent based on the 
        fitted distribution (read-only)."""
        if self._nok_fit is None:
            self._nok_fit = float(
                1 - self.dist.cdf(self.usl, *self.params)
                + self.dist.cdf(self.lsl, *self.params))
        return f'{100 * self._nok_fit:.2f} %'
    
    @property
    def n_errors(self) -> int:
        """Get amount of error values (read-only)."""
        if self._n_errors is None:
            self._n_errors = self.samples.isin(self._error_values).sum()
        return self._n_errors

    @property
    def lsl(self) -> float:
        """Get the lower specification limit (read-only)."""
        return self.spec_limits.lower

    @property
    def usl(self) -> float:
        """Get the upper specification limit (rad-only)."""
        return self.spec_limits.upper

    @property
    def spec_limits(self) -> SpecLimits:
        """Get and set the specification limits."""
        return self._spec_limits
    @spec_limits.setter
    def spec_limits(self, spec_limits: SpecLimits | Specification) -> None:
        if isinstance(spec_limits, Specification):
            spec_limits = spec_limits.LIMITS
        
        if spec_limits.to_tuple() != self.spec_limits.to_tuple():
            self._reset_values_()
        self.spec_limits = spec_limits

    @property
    def control_limits(self) -> Tuple[float, float]:
        """Get lower and upper control limits (read-only)."""
        return (self.lcl, self.ucl)
    
    @property
    def control_range(self) -> float:
        """Get the range (span) of the control limits (read-only)."""
        return self.ucl - self.lcl
    
    @property
    def tolerance(self) -> float:
        """Get tolerance range. If one of the specification limits is 
        not specified, inf is returned (read-only)."""
        return self.spec_limits.tolerance
    
    @property
    def cp(self) -> float | None:
        """Cp is a measure of process capability. Cp is the ratio of the 
        specification width (usl - lsl) to the process variation 
        (agreement*σ). The location is not taken into account by the 
        Cp value. This value therefore only indicates the potential for 
        the Cpk value.
        This value cannot be calculated unless an upper and lower 
        specification limit is given. In this case, None is returned."""
        if self.spec_limits.is_unbounded:
            return None
        
        if self._cp is None:
            self._cp = self.tolerance / (self.agreement * self.std)
        return self._cp
    
    @property
    def cpl(self) -> float:
        """Cpl is a measure of process capability. It is the ratio 
        of the distance between the process mean and the lower 
        specification limit and the lower spread of the process.
        Returns inf if no lower specification limit is specified."""
        return (self.mean - self.lsl) / (self.mean - self.lcl)
    
    @property
    def cpu(self) -> float:
        """Cpu is a measure of process capability. It is the ratio 
        of the distance between the process mean and the upper 
        specification limit and the upper spread of the process.
        Returns inf if no upper specification limit is specified."""
        return (self.usl - self.mean) / (self.ucl - self.mean)
    
    @property
    def cpk(self) -> float | None:
        """Estimates what the process is capable of producing, 
        considering  that the process mean may not be centered between 
        the specification limits. It's calculated as the minimum of
        Cpl and Cpu.
        In general, higher Cpk values indicate a more capable process. 
        Lower Cpk values indicate that the process may need improvement."""
        if self.spec_limits.both_unbounded:
            return None
        return min([self.cpl, self.cpu])
    
    @property
    def Z(self) -> float:
        """The Sigma level Z is another process capability indicator 
        alongside cp and cpk. It describes how many standard deviations 
        can be placed between the mean value and the nearest tolerance 
        limit of a process."""
        if self._Z is None:
            limit = self.lsl if self.cpl < self.cpu else self.usl
            self._Z = abs(self.z_transform(limit))
        return self._Z
    
    @property
    def Z_lt(self) -> float:
        """Statements about long-term capabilities can be derived from 
        short-term capabilities using the σ level. The empirically 
        determined value of 1.5 is subtracted from the σ level."""
        if self._Z_lt is None:
            self._Z_lt = self.Z - SIGMA_DIFFERENCE
        return self._Z_lt

    def mask_error(self) -> 'Series[bool]':
        """Returns a boolean mask indicating which samples are 
        considered errors.

        A sample is marked as an error if its value is found in the 
        predefined set of error values (`_error_values`).

        Returns
        -------
        Series[bool]
            A boolean Series where True indicates an erroneous sample.
        """
        return self.samples.isin(self._error_values)

    def mask_nok(self) -> 'Series[bool]':
        """Returns a boolean mask indicating which filtered samples are 
        out of specification.

        A sample is considered not OK (NOK) if it is less than or equal 
        to the lower specification limit (`lsl`) or greater than or 
        equal to the upper specification limit (`usl`).

        Returns
        -------
        Series[bool]
            A boolean Series where True indicates a sample that is out 
            of spec.
        
        Notes
        -----
        This mask only checks for out-of-spec values and does not 
        consider missing or erroneous samples. Therefore, it is not the 
        exact inverse of `mask_ok()`, which also excludes missing and 
        error values. To get the full set of invalid samples, combine 
        this mask with `mask_missing()` and `mask_error()`.
        """
        return (self.filtered <= self.lsl) | (self.filtered >= self.usl)

    def mask_ok(self) -> Series:
        """Returns a boolean mask indicating which samples are valid 
        (OK).

        A sample is considered OK if it is:
        - Not missing (i.e., not NaN)
        - Not an error (i.e., not in `_error_values`)
        - Within specification limits (`lsl` < value < `usl`)

        Returns
        -------
        Series[bool]
            A boolean Series where True indicates a valid sample.
        
        Notes
        -----
        This mask is the logical inverse of the union of 
        `mask_missing()`, `mask_error()`, and `mask_nok()`. It ensures 
        that only fully valid samples are marked as OK, whereas 
        `mask_nok()` alone does not account for missing or erroneous 
        values.
        """
        return ~(self.mask_missing() | self.mask_error() | self.mask_nok())

    def _reset_values_(self) -> None:
        """Set all values relevant to process capability to None. This 
        function is called when one of the values relevant to the 
        calculation of capability values is adjusted (specification 
        limits or agreement for the control limits). This ensures that 
        the process capability values are recalculated."""
        super()._reset_values_()
        self._n_ok = None
        self._n_nok = None
        self._n_errors = None
        self._ok = None
        self._nok = None
        self._nok_norm = None
        self._nok_fit = None
        self._cp = None
        self._cpk = None
        self._Z = None
        self._Z_lt = None


class GageEstimator(Estimator):
    """Estimates the process capability of a single measurement system
    using the Gage Study Type 1 method.

    Parameters
    ----------
    samples : NumericSample1D
        The samples used to estimate the process capability.
    reference : float | None
        The reference value of the sample under test.
    U_cal : float | None
        The expanded measurement uncertainty of the gage used to 
        measure the reference value.
    tolerance : float | SpecLimits | Specification
        The tolerance range for the measurement.
    resolution : float | None
        The resolution of the measurement system. If None, the 
        resolution is estimated from the samples.
    cg_limit : float, optional
        The limit for the capability index, default is 1.33.
    cgk_limit : float, optional
        The limit for the capability index, default is 1.33.
    tolerance_ratio : float, optional
        The proportion of the tolerance range (between 0 and 1) used 
        to calculate the adjusted limits, default is 0.2 (20%).
    resolution_ratio_limit : float, optional
        The limit for the resolution proportion, default is 0.05.
    n_samples_min : int, optional
        The minimum number of samples required to perform a 
        Measurement System Analysis Type 1, default is 50.
    bias_corrected : bool, optional
        Indicates whether the bias is corrected for the Gage R&R study. 
        If True, the bias is not included in the measurement uncertainty; 
        otherwise, it is included. Default is True.
    
    Examples
    --------

    ```python
    import daspi as dsp

    df = dsp.load_dataset('grnr_adjustment')
    gage = dsp.GageEstimator(
        samples=df['result_gage'],
        reference=df['reference'][0],
        U_cal=df['U_cal'][0],
        tolerance=df['tolerance'][0],
        resolution=df['resolution'][0])
    print(gage.describe())
    ```

    ```console
                       result_gage
    n_samples         6.000000e+01
    n_missing         6.000000e+00
    n_outlier         0.000000e+00
    min               1.000700e+00
    max               1.001600e+00
    R                 9.000000e-04
    mean              1.000809e+00
    median            1.000800e+00
    std               1.306999e-04
    sem               1.778600e-05
    p_ad              9.866969e-20
    lower             9.870000e-01
    upper             1.013000e+00
    cg                3.315484e+01
    cgk               3.109092e+01
    bias              8.092593e-04
    p_bias            3.723292e-44
    T_min_cg          5.214925e-03
    T_min_cgk         1.330752e-02
    T_min_res         2.000000e-03
    resolution_ratio  7.692308e-04
    u_re              2.886751e-05
    u_bias            1.014135e-04
    u_evr             1.306999e-04
    u_ms              1.654302e-04
    ```

    References
    ----------
    [1] Dr. Bill McNeese, BPI Consulting, LLC (09.2012)
        https://www.spcforexcel.com/knowledge/measurement-systems-analysis-gage-rr/anova-gage-rr-part-1/
    
    [2] VDA Band 5, Mess- und Prüfprozesse. Eignung, Planung und Management
        (Juli 2021) 3. überarbeitete Auflage
    """

    __slots__ = (
        '_specification', '_reference', '_U_cal', '_resolution', '_cg_limit', 
        '_cgk_limit', '_tolerance_ratio', '_resolution_ratio_limit', 
        '_n_samples_min', '_cg',  '_cgk', '_limits', '_resolution_ratio', 
        '_bias', '_p_bias', '_T_min_cg', '_T_min_cgk', '_T_min_res', 
        '_process', '_u_re', '_u_bias', '_u_evr', '_u_ms', '_bias_corrected')
    _specification: Specification
    _reference: float
    _U_cal: float
    _resolution: float
    _cg_limit: float
    _cgk_limit: float
    _tolerance_ratio: float
    _resolution_ratio_limit: float
    _n_samples_min: int
    _cg: float | None
    _cgk: float | None
    _limits: SpecLimits | None
    _resolution_ratio: float | None
    _bias: float | None
    _T_min_cg: float | None
    _T_min_cgk: float | None
    _T_min_res: float | None
    _process: ProcessEstimator | None
    _u_re: float | None
    _u_bias: float | None
    _u_evr: float | None
    _u_ms: float | None
    _bias_corrected: bool

    def __init__(self,
            samples: NumericSample1D,
            reference: float | None,
            U_cal: float | None,
            tolerance: float | SpecLimits | Specification,
            resolution: float | None,
            cg_limit: float = 1.33,
            cgk_limit: float = 1.33,
            tolerance_ratio: float = 0.2,
            resolution_ratio_limit: float = 0.05,
            n_samples_min: int = 50,
            bias_corrected: bool = True,
            ) -> None:
        super().__init__(
            samples=samples,
            strategy='norm',
            agreement=6,
            possible_dists=DIST.COMMON,
            evaluate=None)
        self.resolution = resolution
        self.reference = reference
        self.U_cal = U_cal
        self.specification = tolerance
        self._cg_limit = cg_limit
        self._cgk_limit = cgk_limit
        self._tolerance_ratio = tolerance_ratio
        self._resolution_ratio_limit = resolution_ratio_limit
        self._n_samples_min = n_samples_min
        self._bias_corrected = bias_corrected
        if self.n_samples < self._n_samples_min:
            warnings.warn(
                f'The number of samples is too small ({self.n_samples}). '
                f'The minimum number of samples is {self._n_samples_min}.')
        self._descriptive_statistic_attrs = (
            'n_samples',
            'n_missing',
            'n_outlier',
            'min',
            'max',
            'R',
            'mean', 
            'median',
            'std',
            'sem',
            'p_ad',
            'lower',
            'upper',
            'cg',
            'cgk', 
            'bias',
            'p_bias',
            'T_min_cg',
            'T_min_cgk',
            'T_min_res',
            'resolution_ratio',
            'u_re',
            'u_bias',
            'u_evr',
            'u_ms')
        self._reset_values_()
    
    @property
    def resolution(self) -> float:
        """The resolution of the measurement."""
        return self._resolution
    @resolution.setter
    def resolution(self, resolution: float | None) -> None:
        if resolution is None:
            resolution = self.estimate_resolution()
        self._resolution = resolution
        self._reset_values_()
    
    @property
    def reference(self) -> float:
        """The calibrated value of the sample under test."""
        return self._reference
    @reference.setter
    def reference(self, reference: float | None) -> None:
        self._reference = self.median if reference is None else reference
        self._reset_values_()
    
    @property
    def U_cal(self) -> float:
        """The expanded uncertainty of the calibration."""
        return self._U_cal
    @U_cal.setter
    def U_cal(self, U_cal: float | None) -> None:
        self._U_cal = 2 * self.resolution if U_cal is None else U_cal
        self._reset_values_()

    @property
    def specification(self) -> Specification:
        """The specification holding the limits, nominal and tolerance 
        of the process."""
        return self._specification
    @specification.setter
    def specification(
            self, specification: float | SpecLimits | Specification) -> None:
        nominal = self.reference

        if isinstance(specification, Specification):
            tolerance = specification.TOLERANCE
        elif isinstance(specification, SpecLimits):
            tolerance = specification.tolerance
        else:
            tolerance = specification

        self._specification = Specification(nominal=nominal, tolerance=tolerance)
        self._reset_values_()
    
    @property
    def tolerance_ratio(self) -> float:
        """Gets the ratio of the tolerance range used in the calculation 
        of adjusted limits. This value represents the proportion of the 
        total tolerance that is allocated for adjustments."""
        return self._tolerance_ratio
    @tolerance_ratio.setter
    def tolerance_ratio(self, tolerance_ratio: float) -> None:
        self._tolerance_ratio = tolerance_ratio
        self._reset_values_()
    
    @property
    def resolution_ratio_limit(self) -> float:
        """The limit of the resolution ratio."""
        return self._resolution_ratio_limit
    @resolution_ratio_limit.setter
    def resolution_ratio_limit(
            self, resolution_ratio_limit: float) -> None:
        self._resolution_ratio_limit = resolution_ratio_limit
        self._reset_values_()
    
    @property
    def bias_corrected(self) -> bool:
        """Whether the bias is corrected for the Gage R&R study. If 
        True, the bias itself is not included in the measurement 
        uncertainty for the bias; otherwise, it is."""
        return self._bias_corrected
    @bias_corrected.setter
    def bias_corrected(self, bias_corrected: bool) -> None:
        self._bias_corrected = bias_corrected
        self._u_bias = None
    
    @property
    def limits(self) -> SpecLimits:
        """The adjusted specification limits of the process (read-only)."""
        if self._limits is None:
            self._limits = SpecLimits(
                self.nominal - self.tolerance_ratio * self.tolerance / 2,
                self.nominal + self.tolerance_ratio * self.tolerance / 2)
        return self._limits
    
    @property
    def lower(self) -> float:
        """The adjusted lower specification limit (read-only)."""
        return self.limits.lower
    
    @property
    def upper(self) -> float:
        """The adjusted upper specification limit (read-only)."""
        return self.limits.upper
    
    @property
    def cg_limit(self) -> float:
        """The limit of the capability index (read-only)."""
        return self._cg_limit
    
    @property
    def cgk_limit(self) -> float:
        """The limit of the capability index (read-only)."""
        return self._cgk_limit

    @property
    def process(self) -> ProcessEstimator:
        """The process for which the measurement system is to be 
        evaluated (read-only)."""
        if self._process is None:
            self._process = ProcessEstimator(
                samples=self.samples,
                spec_limits=self.limits,
                strategy=self.strategy,
                agreement=self.agreement,
                possible_dists=self.possible_dists)
        return self._process
    
    @property
    def nominal(self) -> float:
        """The nominal value of the specification (read-only)."""
        return self.specification.NOMINAL
    
    @property
    def tolerance(self) -> float:
        """The tolerance of the specification (read-only)."""
        return self.specification.TOLERANCE

    @property
    def tolerance_adj(self) -> float:
        """The adjusted (0.2*T) tolerance of the specification 
        (read-only)."""
        return self.limits.tolerance

    @property
    def n_outlier(self) -> int:
        """The number of outliers in the measurement process (read-only)."""
        return self.process.n_nok
    
    @property
    def cg(self) -> float | None:
        """The repeatability of the measuring system without taking bias
        into account (read-only)."""
        if self._cg is None:
            self._cg = self.process.cp
        return self._cg
    
    @property
    def cgk(self) -> float | None:
        """The repeatability of the measuring system taking into account 
        the bias(read-only)."""
        if self._cgk is None:
            self._cgk = (
                (self.tolerance_adj / 2 - abs(self.bias)) 
                / (self.control_range / 2))
        return self._cgk
    
    @property
    def control_range(self) -> float:
        return self.process.control_range
    
    @property
    def resolution_ratio(self) -> float:
        if self._resolution_ratio is None:
            self._resolution_ratio = self.resolution / self.tolerance
        return self._resolution_ratio
    
    @property
    def bias(self) -> float:
        """The bias of the process (read-only)."""
        if self._bias is None:
            self._bias = self.mean - self.reference
        return self._bias
    
    @property
    def p_bias(self) -> float | None:
        """The probability of the bias being significant by performing
        a t-test (read-only)."""
        if self._p_bias is None:
            self._p_bias = t_test(self.filtered, self.reference)[0]
        return self._p_bias

    @property
    def T_min_cg(self) -> float | None:
        """The minimum allowed tolerance for this testing system based
        on the capability cg of the process (read-only)."""
        if self.cg is None:
            return None
        
        if self._T_min_cg is None:
            self._T_min_cg = (
                self.cg_limit * self.control_range / self.tolerance_ratio)
        return self._T_min_cg
    
    @property
    def T_min_cgk(self) -> float | None:
        """The minimum allowed tolerance for this testing system based
        on the capability cgk of the process (read-only)."""
        if self.cgk is None:
            return None
        
        if self._T_min_cgk is None:
            self._T_min_cgk = (
                (self.cgk_limit * self.control_range / 2 + abs(self.bias))
                / (self.tolerance_ratio / 2))
        return self._T_min_cgk
    
    @property
    def T_min_res(self) -> float:
        """The minimum allowed tolerance for this testing system based
        on the resolution of the testing system (read-only)."""
        if self._T_min_res is None:
            self._T_min_res = (
                self.resolution / self.resolution_ratio_limit)
        return self._T_min_res
    
    @property
    def u_re(self) -> float:
        """The uncertainty of the resolution of the testing system
        (read-only)."""
        if self._u_re is None:
            self._u_re = self.resolution * ANOVA.RESOLUTION_TO_UNCERTAINTY
        return self._u_re
    
    @property
    def u_bias(self) -> float:
        """The uncertainty of the bias of the testing system (read-only)."""
        if self._u_bias is None:
            bias = 0 if self.bias_corrected else self.bias
            self._u_bias = sum(
                map(np.square, (bias, self.U_cal/2, self.std/self.n_samples**0.5))
                )**0.5
        return self._u_bias # type: ignore
    
    @property
    def u_evr(self) -> float:
        """The uncertainty of the expanded variance ratio of the testing
        system (read-only)."""
        if self._u_evr is None:
            self._u_evr = self.std
        return self._u_evr
    
    @property
    def u_ms(self) -> float:
        """The uncertainty of the measurement system (read-only)."""
        if self._u_ms is None:
            u_equipement = max(self.u_re, self.u_evr)
            self._u_ms = sum(
                map(np.square, (self.u_bias, u_equipement))
                )**0.5
        return self._u_ms # type: ignore
    
    def check(self) -> Dict[str, bool]:
        """Perform a few checks to determine if the testing system is 
        capable of measuring the process."""
        checks = dict(
            n_samples=self.n_samples >= 50,
            U_cal=self.U_cal <= self.tolerance * self.tolerance_ratio / 2,
            resolution=self.resolution_ratio <= self.resolution_ratio_limit,
            cg=True if self.cg is None else self.cg >= self.cg_limit,
            cgk=True if self.cgk is None else self.cgk >= self.cgk_limit,)
        return checks
    
    def estimate_resolution(self) -> float:
        """Estimate the resolution of the testing system."""
        return estimate_resolution(self.filtered)
    
    def _reset_values_(self) -> None:
        """Set all values relevant to process capability to None. This 
        function is called when one of the values relevant to the 
        calculation of capability values is adjusted (specification 
        limits or agreement for the control limits). This ensures that 
        the process capability values are recalculated."""
        super()._reset_values_()
        self._limits = None
        self._resolution_ratio = None
        self._bias = None
        self._p_bias = None
        self._cg = None
        self._cgk = None
        self._T_min_cg = None
        self._T_min_cgk = None
        self._T_min_res = None
        self._process = None
        self._u_re = None
        self._u_bias = None
        self._u_evr = None
        self._u_ms = None


def estimate_distribution(
        data: NumericSample1D,
        dists: Tuple[str|rv_continuous, ...] = DIST.COMMON
        ) -> Tuple[rv_continuous, float, Tuple[float, ...]]:
    """First, the p-score is calculated by performing a 
    Kolmogorov-Smirnov test to determine how well each distribution fits
    the data. Whatever has the highest P-score is considered the most
    accurate. This is because a higher p-score means the hypothesis is
    closest to reality.
    
    Parameters
    ----------
    data : NumericSample1D
        1d array of data for which a distribution is to be searched
    dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default DIST.COMMON
    
    Returns
    -------
    dist : scipy.stats rv_continuous
        A generic continous distribution class of best fit
    p : float
        The two-tailed p-value for the best fit
    params : Tuple[float, ...]
        Estimates for any shape parameters (if applicable), followed 
        by those for location and scale. For most random variables, 
        shape statistics will be returned, but there are exceptions 
        (e.g. norm). Can be used to generate values with the help of
        returned dist
    """
    dists = (dists, ) if isinstance(dists, (str, rv_continuous)) else dists
    results = {d: kolmogorov_smirnov_test(data, d) for d in dists}
    dist, (p, _, params) = max(results.items(), key=lambda i: i[1][0])
    dist = ensure_generic(dist)
    return dist, p, params

def _extended_range_(
        data: NumericSample1D,
        margin: float
        ) -> Tuple[float, float]:
    """Returns the extended range of the data. The extended range is
    the range of the data plus a margin. The margin is a percentage
    of the range of the data. The margin is added to the lower and
    upper bound of the data. The formula is as follows:

    $$
    margin_{data} = margin(max(data) - min(dat))
    $$

    $$
    extended_{min} = min(data) - margin_{data}
    $$

    $$
    extended_{max} = max(data) + margin_{data}
    $$
    
    Parameters
    ----------
    data : NumericSample1D
        The data to be used to calculate the extended range
    margin : float
        The margin to be added to the range of the data.
        - 0 returns the range of the data
        - 1 returns 3 times the range: min - 1*range, max + 1*range
    
    Returns
    -------
    Tuple[float, float]
        The extended range of the data
    
    Raises
    ------
    AssertionError:
        If the margin is less than 0
    AssertionError:
        If the margin is too small, so that the extended_max is less or 
        equal the extended_min
    """
    assert margin >= 0, (
        f'Margin must be greater than or equal to 0, got {margin}')

    data_min = min(data)
    data_max = max(data)
    if margin == 0:
        return data_min, data_max
    
    margin_data = margin * (data_max - data_min)
    extended_min = data_min - margin_data
    extended_max = data_max + margin_data
    assert extended_max > extended_min, (
        f'Margin is too small: {margin}')
    
    return extended_min, extended_max

def estimate_kernel_density(
        data: NumericSample1D,
        *,
        stretch: float = 1,
        height: float | None = None, 
        base: float = 0,
        n_points: int = DEFAULT.KD_SEQUENCE_LEN,
        margin: float = 0.5,
        ) -> Tuple[NDArray, NDArray]:
    """Estimates the kernel density of data and returns values that are 
    useful for a plot. If those values are plotted in combination with 
    a histogram, set height as max value of the hostogram.
    
    Kernel density estimation is a way to estimate the probability 
    density function (PDF) of a random variable in a non-parametric way. 
    The used gaussian_kde function of scipy.stats works for both 
    uni-variate and multi-variate data. It includes automatic bandwidth 
    determination. The estimation works best for a unimodal 
    distribution; bimodal or multi-modal distributions tend to be 
    oversmoothed.
    
    Parameters
    ----------
    data : NumericSample1D
        1-D array of datapoints to estimate from.
    stretch : float, optional
        Stretch the distribution estimate by the given factor, is only 
        considered if "height" is None, by default 1
    height : float or None, optional
        If the KDE curve is plotted in combination with other data 
        (e.g. a histogram), you can use height to specify the height at 
        the maximum point of the KDE curve. If this value is specified, 
        the area under the curve will not be normalized, by default None
    base : float, optional
        The curve is shifted in the estimated direction by the given 
        amount. This is usefull for violine plots, by default 0
    n_points : int, optional
        Number of points the estimation and sequence should have,
        by default KD_SEQUENCE_LEN (defined in constants.py)
    margin : float, optional
        Margin for the sequence as factor of data range (max - min ). 
        If margin is 0, The two ends of the estimated density curve then 
        show the minimum and maximum value. Default is 0.

    Returns
    -------
    sequence : 1D array
        Data points at regular intervals from input data minimum to 
        maximum
    estimation : 1D array
        Data points of kernel density estimation
    """
    data = np.array(data)[~np.isnan(data)]
    assert any(data), f'Provided data is empty or contains only zeros: {data}'
    assert any(data != data[0]), f'All provided data have the same value: {data}'
    seq_min, seq_max = _extended_range_(data, margin)
    sequence = np.linspace(seq_min, seq_max, n_points)
    estimation = stats.gaussian_kde(data, bw_method='scott')(sequence)
    stretch = stretch if height is None else height/estimation.max()
    estimation = stretch*estimation + base
    return sequence, estimation

def estimate_kernel_density_2d(
        feature: NumericSample1D,
        target: NumericSample1D,
        *,
        n_points: int = DEFAULT.KD_SEQUENCE_LEN,
        margin: float = 0.5,
        ) -> Tuple[NDArray, NDArray, NDArray]:
    """Estimates the kernel density of 2 dimensional data and returns 
    values that are useful for a contour plot.
    
    Kernel density estimation is a way to estimate the probability 
    density function (PDF) of a random variable in a non-parametric way. 
    The used gaussian_kde function of scipy.stats works for both 
    uni-variate and multi-variate data. It includes automatic bandwidth 
    determination. The estimation works best for a unimodal 
    distribution; bimodal or multi-modal distributions tend to be 
    oversmoothed.
    
    Parameters
    ----------
    feature : NumericSample1D
        A one-dimensional array-like object containing the exogenous
        samples.
    target : NumericSample1D
        A one-dimensional array-like object containing the endogenous 
        samples.
    n_points : int, optional
        Number of points the estimation and sequence should have,
        by default KD_SEQUENCE_LEN (defined in constants.py)
    margin : float, optional
        Margin for the sequence as factor of data range, by default 0.5.

    Returns
    -------
    feature_seq : 2D array
        Data points at regular intervals from input data minimum to 
        maximum used for feature data
    target_seq : 2D array
        Data points at regular intervals from input data minimum to 
        maximum used for target data
    estimation : 2D array
        Data points of kernel density estimation
    
    Raises
    ------
    AssertionError:
        If the provided data is empty, contains only zeros or all values
        are identical.
    """
    feature = np.array(feature)
    target = np.array(target)
    mask = pd.notna(feature) & pd.notna(target)
    target = target[mask]
    feature = feature[mask]
    assert any(target), (
        f'Provided target data is empty or contains only zeros: {target}')
    assert any(feature), (
        f'Provided feature data is empty or contains only zeros: {feature}')
    assert any(target != target[0]), (
        f'All provided target data have the same value: {target}')
    assert any(feature != feature[0]), (
        f'All provided feature data have the same value: {feature}')
    
    f_min, f_max = _extended_range_(feature, margin)
    t_min, t_max = _extended_range_(target, margin)
    feature_seq, target_seq = np.meshgrid(
        np.linspace(f_min, f_max, n_points),
        np.linspace(t_min, t_max, n_points))
    _values = np.vstack([feature, target])
    _sequences = np.vstack([feature_seq.ravel(), target_seq.ravel()])
    estimation = stats.gaussian_kde(_values, bw_method='scott')(_sequences)
    estimation = np.reshape(estimation.T, feature_seq.shape)
    return feature_seq, target_seq, estimation

def estimate_capability_confidence(
        process: ProcessEstimator,
        *,
        kind: Literal['cp', 'cpk'] = 'cpk',
        level: float = 0.95,
        n_groups: int = 1,
        ) -> Tuple[float, float, float]:
    """Calculates the confidence interval for the process capability 
    index (Cp or Cpk) of a process.
    
    This function is an extension of the `cp_ci` and `cpk_ci` functions.
    It instantiates a `ProcessEstimator` and then determines the 
    confidence intervals using the Cp or Cpk values from the estimator.
    
    Parameters
    ----------
    process : ProcessEstimator
        Process Estimator instance, is required to get the necessary 
        process information such as capability indices and number of 
        samples.
    kind : Literal['cp', 'cpk], optional
        Specifies whether to calculate the confidence interval for Cp or 
        Cpk ('cp' or 'cpk'). Defaults is 'cpk'.
    level : float, optional
        The desired confidence level for the interval, expressed as a 
        decimal. Default is 0.95 (95% confidence).
    n_groups : int, optional
        The number of groups for Bonferroni correction to adjust for 
        multiple comparisons. Default is 1, indicating no correction

    Returns
    -------
    Tuple[float, float, float]:
        A tuple containing the estimate, lower bound, and upper bound of 
        the confidence interval for the specified process capability 
        index.
    
    Raises
    ------
    AssertionError:
        If provided kind is not 'cp' or 'cpk'.
    ValueError:
        If no limit is provided or if only one limit is provided and 
        kind is set to 'cp'.
    """
    assert kind in ('cp', 'cpk'), f'Unkown value for {kind=}'
    
    if kind == 'cp':
        if process.cp is None:
            raise ValueError(
                'To calculate the cp values, both limits must be provided')
        ci_values = cp_ci(
            cp=process.cp,
            n_samples=process.n_samples,
            level=level,
            n_groups=n_groups)
    
    elif kind == 'cpk':
        if process.cpk is None:
            raise ValueError(
                'At least one spec limit must be provided')
        ci_values = cpk_ci(
            cpk=process.cpk,
            n_samples=process.n_samples,
            level=level,
            n_groups=n_groups)

    return ci_values

def estimate_resolution(data: NumericSample1D) -> float:
    """Estimate the resolution based on the length of the samples 
    digits.
    
    Parameters
    ----------
    data : NumericSample1D
        1-D array of datapoints to estimate from.

    Returns
    -------
    float
        The estimated resolution.
    """
    n_digits = 0
    for split in map(lambda x: str(x).split('.'), data):
        try:
            _n = len(split[1])
            n_digits = _n if _n > n_digits else n_digits
        except IndexError:
            continue

    return 1 if n_digits == 0 else float(f'0.{"0"*(n_digits-1)}1')


class Kernel:
    """
    Base class for kernel functions used in local weighted regression.

    Kernel functions are used to assign weights to data points based
    on their distance from a target point. The weights decrease as 
    the distance increases, allowing for local modeling of data.
    
    Parameters
    ----------
    bandwidth : float
        The bandwidth parameter that determines the locality of the 
        smoothing. It controls how far from the target point `x_i` the 
        weights will be considered.
    """

    bandwidth: float
    """The bandwidth parameter that determines the locality of the 
    smoothing. It controls how far from the target point the weights 
    will be considered. A smaller bandwidth results in  more localized 
    weights, while a larger bandwidth considers a wider range of points."""

    def __init__(self, bandwidth: float) -> None:
        self.bandwidth = bandwidth

    @abstractmethod
    def weights(self, x_i: float, x: NumericSample1D) -> NDArray:
        """
        Calculate weights for a target point x_i based on input x.

        Parameters
        ----------
        x_i : float
            The target point for which the weights are being calculated.
        x : Any
            The input data points. This can be a 1-dimensional array or 
            list of values.

        Returns
        -------
        np.ndarray
            An array of weights corresponding to the input data points.
            The specific weighting strategy is defined in the subclasses.
        
        Raises
        ------
        NotImplementedError
            If this method is not overridden by a subclass.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses.")


class TriCubeKernel(Kernel):
    """Tri-cube kernel function.
    
    The tricube kernel is used to assign weights to data points based on 
    their distance from a target point. The weights decrease as the 
    distance increases, and points further away than the specified 
    bandwidth receive zero weight.
    
    Parameters
    ----------
    bandwidth : float
        The bandwidth parameter that determines the locality of the 
        smoothing. It controls how far from the target point `x_i` the 
        weights will be considered.
    """

    def weights(self, x_i: float, x: NumericSample1D) -> NDArray:
        """Calculate weights for a target point x_i based on input x.

        Parameters
        ----------
        x_i : float
            The target point for which the weights are being calculated.
        x : NumericSample1D
            The input data points. This can be a 1-dimensional array or 
            list of values.

        Returns
        -------
        weights : NDArray
            An array of weights corresponding to the input data points. 
            The weights are computed using the tricube kernel, where 
            points closer to `x_i` receive higher weights and points 
            further away receive lower weights, tapering to zero beyond 
            the specified bandwidth.

        Notes
        -----
        The tricube weight function gives points closer to zero higher 
        weights and smoothly decreases to zero at x = ±1. This creates a 
        smooth weighting scheme for local regression.
        The tricube kernel is defined as:

            w(u) = (1 - u^3)^3 for |u| <= 1
            w(u) = 0 for |u| > 1

        where u is the normalized distance calculated as:

            u = |(x - x_i) / bandwidth|
        """
        u = np.abs((np.asarray(x) - x_i) / self.bandwidth)
        weights = np.where(u <= 1, (1 - u**3)**3, 0)
        return weights


class EpanechnikovKernel(Kernel):
    """Epanechnikov kernel function.
    
    The Epanechnikov kernel is used to assign weights to data points 
    based on their distance from a target point. It is a parabolic 
    kernel that gives maximum weight to points at the center and 
    decreases towards the edges.
    
    Parameters
    ----------
    bandwidth : float
        The bandwidth parameter that determines the locality of the 
        smoothing. It controls how far from the target point `x_i` the 
        weights will be considered."""

    def weights(self, x_i: float, x: NumericSample1D) -> NDArray:
        """Calculate weights for a target point x_i based on input x.

        Parameters
        ----------
        x_i : float
            The target point for which the weights are being calculated.
        x : array_like
            The input data points. This can be a 1-dimensional array or 
            list of values.

        Returns
        -------
        weights : ndarray
            An array of weights corresponding to the input data points. 
            The weights are computed using the Epanechnikov kernel, 
            where points closer to `x_i` receive higher weights, 
            tapering to zero beyond the specified bandwidth.

        Notes
        -----
        The Epanechnikov kernel is defined as:

            w(u) = 0.75 * (1 - u^2) for |u| <= 1
            w(u) = 0 for |u| > 1

        where u is the normalized distance calculated as:

            u = |(x - x_i) / bandwidth|
        """
        u = np.abs((np.asarray(x) - x_i) / self.bandwidth)
        weights = np.where(u <= 1, 0.75 * (1 - u**2), 0)
        return weights


class GaussianKernel(Kernel):
    """Gaussian kernel function.
    
    The Gaussian kernel is used to assign weights to data points based 
    on their distance from a target point. It is a smooth kernel that 
    gives non-zero weights to all points, with the weights decreasing 
    exponentially as the distance increases.
    
    Parameters
    ----------
    bandwidth : float
        The bandwidth parameter that determines the locality of the 
        smoothing. It controls how far from the target point `x_i` the 
        weights will be considered.
    """
    def weights(self, x_i: float, x: NumericSample1D) -> NDArray:
        """Calculate weights for a target point x_i based on input x.

        Parameters
        ----------
        x_i : float
            The target point for which the weights are being calculated.
        x : NumericSample1D
            The input data points. This can be a 1-dimensional array or list of values.
        bandwidth : float
            The bandwidth parameter that determines the locality of the smoothing. 
            It controls how far from the target point `x_i` the weights will be considered.

        Returns
        -------
        weights : ndarray
            An array of weights corresponding to the input data points. The weights are 
            computed using the Gaussian kernel, where all points receive a non-zero weight 
            that decreases with distance from `x_i`.

        Notes
        -----
        The Gaussian kernel is defined as:

            w(u) = (1 / sqrt(2 * pi)) * exp(-0.5 * u^2)

        where u is the normalized distance calculated as:

            u = (x - x_i) / bandwidth
        """
        u = (np.asarray(x) - x_i) / self.bandwidth
        weights = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        return weights

class TukeyBiweightKernel(Kernel):
    """Tukey's biweight kernel function.
    
    The Tukey biweight kernel is used to assign weights to data points 
    based on their distance from a target point. It gives maximum weight 
    to points at the center and decreases towards the edges, tapering 
    off completely beyond the specified bandwidth. This kernel is robust 
    to outliers.

    Parameters
    ----------
    bandwidth : float
        The bandwidth parameter that determines the locality of the 
        smoothing. It controls how far from the target point `x_i` the 
        weights will be considered.
    """

    def weights(self, x_i: float, x: NumericSample1D) -> NDArray:
        """Calculate weights for a target point x_i based on input x.

        Parameters
        ----------
        x_i : float
            The target point for which the weights are being calculated.
        x : NumericSample1D
            The input data points. This can be a 1-dimensional array or 
            list of values.

        Returns
        -------
        weights : NDArray
            An array of weights corresponding to the input data points. 
            The weights are computed using Tukey's biweight kernel, 
            where points closer to `x_i` receive higher weights, 
            tapering to zero beyond the specified bandwidth.

        Notes
        -----
        The Tukey biweight kernel is defined as:

            w(u) = 0.5 * (1 - u^2)^2 for |u| <= 1
            w(u) = 0 for |u| > 1

        where u is the normalized distance calculated as:

            u = |(x - x_i) / bandwidth|
        """
        u = np.abs((np.asarray(x) - x_i) / self.bandwidth)
        weights = np.where(u <= 1, 0.5 * (1 - u**2)**2, 0)
        return weights


class Loess:
    """Smooth the data using Locally Estimated Scatterplot Smoothing 
    (LOESS).
    
    LOESS is not necessarily suitable for all regression models due to 
    its non-parametric approach and high computational intensity. 
    Nonetheless, it serves as an effective method for modeling the 
    relationship between two variables that do not adhere to a 
    predefined distribution and exhibit a non-linear relationship.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''
    fit_at_init : bool, optional
        Whether to fit the model at initialization, by default True
    **kwds
        Keyword arguments for the `fit` method. Is only taken into 
        account if `fit_at_init` is True.
    
    Examples
    --------
    ``` python
    import numpy as np
    import daspi as dsp
    import pandas as pd
    import matplotlib.pyplot as plt

    x = 5*np.random.random(100)
    data = pd.DataFrame(dict(
        x = x,
        y = np.sin(x) * 3*np.exp(-x) + np.random.normal(0, 0.2, 100)))
    model = dsp.Loess(data, 'y', 'x')
    sequence, prediction, lower, upper = model.fitted_line(0.95)

    fig, ax = plt.subplots()
    ax.scatter(data.x, data.y)
    ax.plot(sequence, prediction)
    ax.fill_between(sequence, lower, upper, alpha=0.2)
    ```
    
    Sources
    -------
    [1] James Brennan (2020), Confidence intervals for LOWESS models in 
    python [https://james-brennan.github.io](https://james-brennan.github.io/posts/lowess_conf/)

    [2] Soul Dobilas (2020), LOWESS Regression in Python: How to 
    Discover Clear Patterns in Your Data?, 
    [Towards Data Science](https://towardsdatascience.com/lowess-regression-in-python-how-to-discover-clear-patterns-in-your-data-f26e523d7a35)
    
    [3] Btyner (2006), Local Regression [Wikipedia](https://en.wikipedia.org/w/index.php?title=Local_regression&oldid=1261263154)
    """
    __slots__ = (
        'source', 'target', 'feature', 'smoothed', 'std_errors', 'fraction',
        'kernel', 'order')
    
    source: DataFrame
    """The data source for the plot"""
    target: str
    """The column name of the target variable."""
    feature: str
    """The column name of the feature variable."""
    smoothed: Series
    """Smoothed target data as a pandas Series."""
    std_errors: Series
    """Standard errors of the smoothed target data as a pandas Series."""
    fraction: float
    """The fraction of data points used in each local regression."""
    kernel: Kernel
    """The kernel for weights function used in the LOESS smoothing."""
    order: Literal[0, 1, 2, 3]
    """The order of local regression:

    - 0: No smoothing (interpolation)
    - 1: Linear regression
    - 2: Quadratic regression
    - 3: Cubic regression
    
    The order determines the degree of the polynomial used in the local 
    regression, affecting the flexibility of the fitted curve.
    High-degree polynomials would tend to overfit the data in each 
    subset and are numerically unstable, making accurate computations 
    difficult."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            *,
            fit_at_init: bool = True,
            **kwds
            ) -> None:
        source = (source
            .copy()
            [[feature, target]]
            .dropna(axis=0, how='any')
            .sort_values(feature)
            .reset_index(drop=True))
        assert not source.empty, (
            'No data after removing missing values.')
        self.source = source
        self.target = target
        self.feature = feature
        if fit_at_init:
            self.fit(**kwds)
    
    @property
    def x(self) -> Series:
        """The feature variable as exogenous variable (read-only)."""
        return self.source[self.feature]
    
    @property
    def y(self) -> Series:
        """The target variable as endogenous variable (read-only)."""
        return self.source[self.target]
    
    @property
    def fitted(self) -> bool:
        """True if the data has been fitted (read-only)."""
        return hasattr(self, 'smoothed')

    @property
    def n_samples(self) -> int:
        """Amount of samples in source after removing missing values
        (read-only)."""
        return len(self.y)
    
    @property
    def residuals(self) -> Series:
        """Residuals as difference between target and lowess (read-only)."""
        self._check_fitted_('residuals')
        return self.y - self.smoothed
    
    @property
    def dof_resid(self) -> int:
        """Degree of freedom of residuals (read-only)."""
        self._check_fitted_('dof_resid')
        return self.n_samples - self.order - 1
    
    @property
    def available_kernels(self) -> Dict[str, Type[Kernel]]:
        """Available kernels for smoothing (read-only)."""
        kernels = {
            'tricube': TriCubeKernel,
            'gaussian': GaussianKernel,
            'epanechnikov': EpanechnikovKernel}
        return kernels
    
    def bandwidth(self, fraction: float) -> float:
        """Get the bandwidth parameter that determines the locality of 
        the smoothing. It controls how far from the target point the 
        weights will be considered. A larger bandwidth results in 
        smoother fits, while a smaller bandwidth allows for more local 
        variation to be captured in the fitted curve."""
        return fraction * (self.x.max() - self.x.min())
    
    def _check_fitted_(self, method_name: str) -> None:
        """Check if the model has been fitted.
        
        Raises
        ------
        AssertionError
            If the model has not been fitted yet using the fit method.
        """
        assert self.fitted, (
            f'The data must be fitted before {method_name} can be performed.')

    def _interpolate_(self, **kwds) -> Callable:
        """Get a function using the `interp1d` method from 
        `scipy.interpolate` by passing the feature and target values 
        from the LOWESS output. This function can be used to take new 
        feature values and generate target values them.
        
        Returns
        -------
        Callable:
            A callable function that can be used to interpolate new
            target values from feature values.
        """
        self._check_fitted_('interpolate')
        _kwds: Dict[str, Any] = dict(
            x=self.x,
            y=self.smoothed,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True
            ) | kwds
        return interp1d(**_kwds)

    def _wls_(
            self,
            W: NDArray
            ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        X = np.vstack([self.x**i for i in range(self.order + 1)]).T
        XtW = X.T @ W           # (n_beta x n) @ (n x n) = (n_beta x n)
        XtWX = XtW @ X          # (n_beta x n) @ (n x n_beta) = (n_beta x n_beta)
        XtWy = XtW @ self.y     # (n_beta x n) @ (n x 1) = (n_beta x 1)
        try:
            XtWX_inv = np.linalg.inv(XtWX)
            beta = XtWX_inv @ (XtWy) # (n_beta x n_beta) @ (n_beta x 1) = (n_beta x 1)
        except LinAlgError:
            raise LinAlgError(
                'Matrix is singular or close to singular, cannot compute '
                'inverse. To avoid this error, try increasing the fraction or '
                'changing the kernel to Gaussian. For more information, see '
                'the Notes section in the Fit method docstring.')
        return beta, X, XtW, XtWX_inv

    def _smoothed_value_(self, row: Series) -> float:
        """Get the smoothed value of the target variable at a given
        feature value."""
        index = int(row.name) # type: ignore
        x_i = row[self.feature]
        weights = self.kernel.weights(x_i, self.x)
        W = np.diag(weights)
        beta, X, XtW, XtWX_inv = self._wls_(W)
        fitted_values = X @ beta
        eta = (self.y - fitted_values).to_numpy()
        self.std_errors[index] = self._calculate_standard_error_(
            index, eta, X, W, XtW, XtWX_inv)
        return fitted_values[index]

    def _calculate_standard_error_(
            self,
            index: int,
            eta: NDArray,
            X: NDArray,
            W: NDArray,
            XtW: NDArray,
            XtWX_inv: NDArray,) -> float:
        """Calculate the standard error for the fitted value at a given 
        index. First calculate leverage for current point, then
        calculate standard error incorporating leverage"""
        H = X @ XtWX_inv @ XtW # (n_beta x n) @ (n_beta x n_beta) @ (n_beta x n) = (n x n)
        leverage = H[index, index]
        sigma2 = np.sum(W @ eta**2) / (np.trace(W) - (self.order + 1))
        return np.sqrt(sigma2 * leverage)
    
    def fit(
            self,
            fraction: float = 0.6,
            order: Literal[0, 1, 2, 3] = 3,
            kernel: Literal['tricube', 'gaussian', 'epanechnikov'] = 'tricube'
            ) -> Self:
        """Fits the model using the `statsmodels.nonparametric.lowess` 
        method.

        Parameters
        ----------
        fraction : float, optional
            The fraction of the data used for each local regression. A 
            good value to start with is > 1/2 (default value of 
            statsmodels is 2/3). Reduce the value to avoid underfitting. 
            A value below 0.2 usually leads to overfitting exept for 
            gaussian weights. Default is 0.6
        order : Literal[0, 1, 2, 3], optional
            The order of the local regression to be fitted. 
            This determines the degree of the polynomial used in the 
            local regression:
            - 0: No smoothing (interpolation)
            - 1: Linear regression
            - 2: Quadratic regression
            - 3: Cubic regression
            Default is 3.
        kernel : Literal['tricube', 'gaussian', 'epanechnikov'], optional
            The kernel function used to calculate the weights. 
            Available kernels are:
            - 'tricube': Tricube kernel function
            - 'gaussian': Gaussian kernel function
            - 'epanechnikov': Epanechnikov kernel function
            Default is 'tricube'.

        Returns
        -------
        Lowess:
            The instance of the Lowess with the fitted values.
        
        Notes
        -----
        If using this method it's possible to run in a LinAlgError.
        This error usually happens in two main scenarios:

        1. Singular Matrix: 
            When the input data creates a singular matrix 
            (determinant = 0). This often occurs when:
            - You have perfectly correlated features
            - You have duplicate data points
            - There's not enough variation in your data

        2. Ill-Conditioned Matrix:
            When the matrix is nearly singular. Common causes:
            - Features with very different scales
            - Multicollinearity between features
        """
        assert kernel in self.available_kernels, (
            f'The specified kernel "{kernel}" is not recognized. '
            f'Available are: {", ".join(self.available_kernels.keys())}.')
        assert order in (0, 1, 2, 3), (
            f'Local regression order must be 1, 2, 3 or 4, not {order}.')
        
        self.order = order
        Kernel_ = self.available_kernels[kernel]
        self.kernel = Kernel_(self.bandwidth(fraction))
        self.std_errors = pd.Series(np.zeros(self.n_samples))
        self.smoothed  = self.source.apply(self._smoothed_value_, axis=1)
        return self
    
    def predict(
        self,
        x: int | float | NumericSample1D,
        kind: str | int = 'linear') -> NDArray:
        """Predict the target value(s) for the given feature value(s).

        Parameters
        ----------
        x : int | float | NumericSample1D
            The feature value(s) for which to predict the target 
            value(s).
        kind : str or int, optional
            Specifies the kind of interpolation as a string or as an 
            integer specifying the order of the spline interpolator to 
            use. The string has to be one of 'linear', 'nearest', 
            'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 
            'previous', or 'next'.
            - **'zero', 'slinear', 'quadratic' and 'cubic':** refer to a 
              spline interpolation of zeroth, first, second or third 
              order.
            - **'previous' and 'next':** simply return the previous or 
              next value of the point
            - **'nearest-up' and 'nearest':** differ when interpolating 
            half-integers (e.g. 0.5, 1.5) in that 'nearest-up' rounds up 
            and 'nearest' rounds down. Default is 'linear'.
        
        Returns
        -------
        NDArray
            The predicted target value(s) for the given feature value(s)
            as one-dimensonal numpy array.
        """
        self._check_fitted_('predict')
        if isinstance(x, (int, float)):
            x = [x]
        return self._interpolate_(kind=kind)(x)
    
    @overload
    def fitted_line(self) -> Tuple[NDArray, NDArray]: ...
    @overload
    def fitted_line(
            self, confidence_level: None, n_points: int = ...
            ) -> Tuple[NDArray, NDArray]: ...
    @overload
    def fitted_line(
            self, confidence_level: float, n_points: int = ...
            ) -> Tuple[NDArray, NDArray, NDArray, NDArray]: ...

    def fitted_line(
            self,
            confidence_level: float | None = None,
            n_points: int = DEFAULT.LOWESS_SEQUENCE_LEN
            ) -> Any:
        """Generate a smooth sequence of predictions from the fitted 
        LOWESS model.
        
        This method creates an evenly spaced sequence of feature values 
        and predicts their corresponding target values using linear 
        interpolation. It's particularly useful for:
        1. Plotting smooth trend lines
        2. Visualizing the LOWESS fit
        3. Generating continuous predictions across the feature range
        
        Parameters
        ----------
        confidence_level : float | None, optional
            If provided, calculate confidence bands at this level 
            (0 to 1).
            Example: 0.95 for 95% confidence bands. If None, no 
            confidence bands are calculated. Default is None.
        n_points : int, optional
            Number of points to generate in the sequence. More points 
            create a smoother visualization but increase computation 
            time. Default is defined in DEFAULT.LOWESS_SEQUENCE_LEN.
        
        Returns
        -------
        sequence : NDArray
            Evenly spaced feature values
        prediction : NDArray
            Predicted target values
        lower_band : NDArray, optional
            Lower confidence band. Only returned if confidence_level is 
            provided
        upper_band : NDArray, optional
            Upper confidence band. Only returned if confidence_level is 
            provided
        """
        # Generate sequence and predictions
        sequence = np.linspace(self.x.min(), self.x.max(), n_points)
        prediction = self.predict(sequence, kind='linear')
        
        if confidence_level is None:
            return sequence, prediction
        else:
            alpha = confidence_to_alpha(confidence_level, two_sided=True)
            t_ppf = stats.t.ppf(1 - alpha, self.dof_resid)
            se_interpolated = self._interpolate_(y=self.std_errors)(sequence)
            margin = t_ppf * se_interpolated
            lower_band = prediction - margin
            upper_band = prediction + margin
            
            return sequence, prediction, lower_band, upper_band


class Lowess(Loess):
    """Smooth the data using Locally Weighted Scatterplot Smoothing (LOWESS).
    
    LOWESS is a non-parametric regression method that combines multiple 
    regression models in a k-nearest-neighbor-based meta-model. It is 
    particularly useful for fitting a smooth curve to data that may 
    exhibit non-linear relationships. The robustness weights are used to 
    reduce the influence of outliers in the data. They are calculated 
    based on the residuals from the previous iteration of the smoothing 
    process.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot, by default ''
    
    Examples
    --------
    ``` python
    import numpy as np
    import daspi as dsp
    import pandas as pd
    import matplotlib.pyplot as plt

    x = 5*np.random.random(100)
    data = pd.DataFrame(dict(
        x = x,
        y = np.sin(x) * 3*np.exp(-x) + np.random.normal(0, 0.2, 100)))
    model = dsp.Lowess(data, 'y', 'x')
    sequence, prediction, lower, upper = model.fitted_line(0.95)

    fig, ax = plt.subplots()
    ax.scatter(data.x, data.y)
    ax.plot(sequence, prediction)
    ax.fill_between(sequence, lower, upper, alpha=0.2)
    ```
    """    
    __slots__ = ('robustness_kernel')
    
    robustness_kernel: TukeyBiweightKernel
    """The kernel function used to assign weights for adjusting the 
    LOESS fitted values, providing robustness against outliers."""

    def fit(
            self,
            fraction: float = 0.6,
            order: Literal[0, 1, 2, 3] = 3,
            kernel: Literal['tricube', 'gaussian', 'epanechnikov'] = 'tricube'
            ) -> Self:
        self.robustness_kernel = TukeyBiweightKernel(self.bandwidth(fraction))
        return super().fit(fraction=fraction, kernel=kernel, order=order)

    def _smoothed_value_(self, row: Series) -> float:
        """Get the smoothed value of the target variable at a given
        feature value, applying robustness to reduce the influence of outliers."""
        index = int(row.name)  # type: ignore
        x_i = row[self.feature]
        weights = self.kernel.weights(x_i, self.x)
        W = np.diag(weights)
        # First pass to get initial fitted values
        beta, X, XtW, XtWX_inv = self._wls_(W)
        _fitted_values = X @ beta

        # Update the robustness kernel weights based on the residuals
        # Then perform local weighted regression again with robust weights
        _eta = (self.y - _fitted_values).to_numpy()
        y_hat_i = _fitted_values[index]
        robust_weights = self.robustness_kernel.weights(y_hat_i, _eta)
        Wr = np.diag(robust_weights)
        beta_robust, X, XtWr, XtWrX_inv = self._wls_(Wr)
        
        # Calculate the smoothed value using robust fitted values
        fitted_values = X @ beta_robust
        eta = (self.y - fitted_values).to_numpy()
        self.std_errors[index] = self._calculate_standard_error_(
            index, eta, X, W, XtW, XtWX_inv)
        
        return fitted_values[index]


__all__ = [
    'Estimator',
    'ProcessEstimator',
    'GageEstimator',
    'estimate_distribution',
    'estimate_kernel_density',
    'estimate_kernel_density_2d',
    'estimate_capability_confidence',
    'estimate_resolution',
    'Loess',
    'Lowess']
