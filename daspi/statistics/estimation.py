# source for ci: https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#conf_int_of_var

import numpy as np
import pandas as pd

from typing import List
from typing import Tuple
from typing import Literal
from typing import Optional
from numpy.typing import ArrayLike
from scipy.stats._distn_infrastructure import rv_continuous

from scipy import stats
from scipy.stats import sem
from scipy.stats import skew
from scipy.stats import kurtosis

from .hypothesis import skew_test
from .hypothesis import kurtosis_test
from .hypothesis import variance_stability_test
from .hypothesis import kolmogorov_smirnov_test
from .._constants import KDE
from .._constants import DISTRIBUTION


class Estimator:

    __slots__ = (
        '_data', '_filtered', '_n_samples', '_n_missing', '_mean', '_median', 
        '_std', '_excess', '_p_excess', '_skew', '_p_skew', '_dist', '_p_dist',
        '_dist_params', 'possible_dists')
    _data: pd.Series
    _filtered: pd.Series
    _n_samples: int | None
    _n_missing: int | None
    _mean: float | None
    _median: float | None
    _std: float | None
    _excess: float | None
    _p_excess: float | None
    _skew: float | None
    _p_skew: float | None
    _dist: rv_continuous | None
    _p_dist: float | None
    _dist_params: tuple | None
    possible_dists: Tuple[str | rv_continuous]

    def __init__(
            self, data: ArrayLike, 
            possible_dists: Tuple[str | rv_continuous] = DISTRIBUTION.COMMON
            ) -> None:
        self._n_samples = len(data)
        self._n_missing = None
        self._mean = None
        self._median = None
        self._std = None
        self._excess = None
        self._p_excess = None
        self._skew = None
        self._p_skew = None
        self._dist = None
        self._p_dist = None
        self._dist_params = None
        self.possible_dists = possible_dists
        self._filtered = pd.Series()
        self._data = data if isinstance(data, pd.Series) else pd.Series(data)
        
    @property
    def data(self) -> pd.Series:
        """Get the raw data as it was given during instantiation as
        pandas Series."""
        return self._data
        
    @property
    def filtered(self) -> pd.Series:
        """Get the data without error values and no missing value"""
        if self._filtered.empty:
            self._filtered = self.data[self.data.notna()]
        return self._filtered
    
    @property
    def n_samples(self):
        """Get sample size of unfiltered data"""
        return self._n_samples
    
    @property
    def n_missing(self):
        """Get amount of missing values"""
        if self._n_missing is None:
            self._n_missing = self.data.isna().sum()
        return self._n_missing

    @property
    def mean(self) -> float:
        """Get mean of filtered data"""
        if self._mean is None:
            self._mean = self.filtered.mean()
        return self._mean

    @property
    def median(self) -> float:
        """Get median of filtered data"""
        if self._median is None:
            self._median = self.filtered.median()
        return self._median
    
    @property
    def std(self) -> float:
        """Get standard deviation of filtered data"""
        if self._std is None:
            self._std = self.filtered.std()
        return self._std
    
    @property
    def excess(self) -> float:
        """Get the Fisher kurtosis (excess) of the filtered data.
        Calculations are corrected for statistical bias.
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
            self._excess = kurtosis(self.filtered, fisher=True, bias=False)
        return self._excess
    
    @property
    def p_excess(self) -> float:
        """Get the probability that the excess of the population that
        the sample was drawn from is the same as that of a corresponding
        normal distribution."""
        if self._p_excess is None:
            self._p_excess = kurtosis_test(self.filtered)[0]
        return self._p_excess
    
    @property
    def skew(self) -> Tuple[float]:
        """Get the sample skewness of the filtered data.
        Calculations are corrected for statistical bias.
        For normally distributed data, the skewness should be about zero. 
        For unimodal continuous distributions, a skewness value greater 
        than zero means that there is more weight in the right tail of 
        the distribution:
            - skew < 0: left-skewed -> long tail left
            - skew > 0: right-skewed -> long tail right
        """
        if self._skew is None:
            self._skew = skew(self.filtered, bias=False)
        return self._skew
    
    @property
    def p_skew(self) -> float:
        """Get the probability that the skewness of the population that
        the sample was drawn from is the same as that of a corresponding
        normal distribution"""
        if self._p_skew is None:
            self._p_skew = skew_test(self.filtered)[0]
        return self._p_skew

    @property
    def dist(self) -> rv_continuous | None:
        """Get fitted distribution. None if method distribution has 
        not been called."""
        return self._dist
    
    @property
    def p_dist(self) -> float | None:
        """Get probability of fitted distribution. None if method 
        distribution  has not been called."""
        return self._p_dist
    
    @property
    def dist_params(self) -> tuple:
        """Get params of fitted distribution. None if method 
        distribution has not been called."""
        return self._dist_params
    
    def distribution(self):
        """First, the p-score is calculated by performing a 
        Kolmogorov-Smirnov test to determine how well each distribution fits
        the data. Whatever has the highest P-score is considered the most
        accurate. This is because a higher p-score means the hypothesis is
        closest to reality.""" # TODO link docstring
        self._dist, self._p_dist, self._dist_params = estimate_distribution(
            self.filtered, self.possible_dists)
    
    def stable_variance(
            self, alpha: float = 0.05, n_sections : int = 3) -> bool:
        """Test whether the variance remains stable across the data. 
        
        The data is divided into 5 subgroups and the variances of their 
        sections are checked using the Levene test.
        
        Parameters
        ----------
        alpha : float
            Alpha risk of hypothesis tests. If a p-value is below this 
            limit, the null hypothesis is rejected
        n_sections : int, optional
            Amount of sections to divide the data into, by default 5
        
        Returns
        -------
        stable : bool
            True if the p-value > alpha
        """
        assert isinstance(n_sections, int)
        assert  1 < n_sections < len(self.filtered)
        p, L = variance_stability_test(self.filtered, n_sections=n_sections)
        return p > alpha
    
    def follows_norm_curve(self, alpha: float = 0.05) -> bool:
        """Two sided hypothesis whether the skew or the excess 
        (kurtosis) is different from the normal distribution.
        
        Parameters
        ----------
        alpha : float
            Alpha risk of hypothesis tests. If a p-value is below this 
            limit, the null hypothesis is rejected
            
        Returns
        -------
        remain_h0 : bool
            True if both p-values > alpha, otherwise False
        """
        remain_h0 = [
            self.p_excess > alpha,
            self.p_skew > alpha]
        return all(remain_h0)


class ProcessEstimator(Estimator):

    __slots__ = (
        '_lsl', '_usl', '_n_ok', '_n_nok', '_error_values', '_n_errors', 
        '_cp', '_cpk', '_control_limits', '_tolerance', '_q_low', '_q_upp')
    lsl: float | None
    usl: float | None
    _n_ok: int | None
    _n_nok: int | None
    _error_values: Tuple[float]
    _n_errrors: int
    _cp: int | None
    _cpk: int | None
    _control_limits: str
    _tolerance: int
    _q_low: float | None
    _q_upp: float | None

    def __init__(
            self, data: ArrayLike,
            lsl: Optional[float] = None, 
            usl: Optional[float] = None, 
            error_values: Tuple[float] = (),
            control_limits : Literal['eval', 'fit', 'norm', 'data'] = 'eval',
            tolerance: float | int = 6, 
            possible_dists: Tuple[str | rv_continuous] = DISTRIBUTION.COMMON
            ) -> None:
        """"
        Parameters
        ----------
        data : array like
            1D array of process data.
        lsl, usl : float
            Lower and upper specification limit for process data.
        error_values : tuple of float, optional
            If the process data may contain coded values for measurement 
            errors or similar, they can be specified here, 
            by default [].
        control_limits : {'eval', 'fit', 'norm', 'data'}, optional
            Which strategy should be used to determine the control 
            limits for process variation:
            - eval: The strategy is determined according to the given 
            flowchart
            - fit: First, the distribution is searched for that best 
            represents the process data and then the process variation 
            tolerance is calculated
            - norm: it is assumed that the data is subject to normal 
            distribution. The variation tolerance is then calculated as 
            tolerance * standard deviation
            - data: The quantiles for the process variation tolerance 
            are read directly from the data.
        tolerance : float or int, optional
            Specify the tolerated process variation for which the 
            control limits are to be calculated. 
            - If int, the spread is determined using the normal 
            distribution tolerance*sigma, 
            e.g. tolerance = 6 -> 6*sigma ~ covers 99.75% of the data. 
            The upper and lower permissible quantiles are then 
            calculated from this.
            - If float, the value must be between 0 and 1. This value is 
            then used directly to calculate the upper and lower 
            allowable quantiles
            by default 6
        possible_dists : tuple of strings or rv_continous, optional
            Distributions to which the data may be subject. Only 
            continuous distributions of scipy.stats are allowed,
            by default DISTRIBUTION.COMMON

        """# TODO link docstring
        assert usl > lsl if None not in (lsl, usl) else True
        if isinstance(tolerance, int): assert tolerance > 1
        if isinstance(tolerance, float): assert 0 < tolerance < 1
        assert control_limits in ['eval', 'fit', 'norm', 'data']
        self._n_ok = None
        self._n_nok = None
        self._n_errors = None
        self._lcl = None
        self._ucl = None
        self._cp = None
        self._cpk = None
        self._q_low = None
        self._q_upp = None
        self._error_values = error_values
        self._lsl = lsl
        self._usl = usl
        self._control_limits = control_limits
        self.tolerance = tolerance
        self._reset_capabilty_values_()
        super().__init__(data, possible_dists)
        
    @property
    def filtered(self) -> pd.Series:
        """Get the data without error values and no missing value"""
        if self._filtered.empty:
            self._filtered = self.data[
                ~self.data.isin(self._error_values)
                & self.data.notna()]
        return self._filtered
    
    @property # TODO
    def n_ok(self) -> int:
        if self._n_ok is None:
            self._n_ok = (self.n_samples
                          - self.n_nok
                          - self.n_errors
                          - self.n_missing)
        return self._n_ok
    
    @property # TODO
    def n_nok(self) -> int:
        if self._n_nok is None:
            self._n_nok = (
                (self.filtered >= self.usl).sum() if self.usl else 0
                + (self.filtered <= self.lsl).sum() if self.lsl else 0)
        return self.n_nok
    
    @property
    def p_ok(self):
        """Get amount of OK-values as percent"""
        return 100*self.n_ok/self.n_samples
    
    @property
    def p_nok(self):
        """Get amount of NOK-values as percent"""
        return 100*self.n_nok/self.n_samples
    
    @property # TODO
    def n_errors(self) -> int:
        if self._n_errors is None:
            self._n_errors = self.data.isin(self._error_values).sum()
        return self._n_errors

    @property
    def lsl(self):
        """Get lower specification limit"""
        return self._lsl
    @lsl.setter
    def lsl(self, lsl: float):
        self._lsl = lsl
        self._reset_capabilty_values_()

    @property
    def usl(self):
        """Get upper specification limit"""
        return self._lsl
    @usl.setter
    def usl(self, usl: float):
        self._usl = usl
        self._reset_capabilty_values_()

    @property
    def q_low(self) -> float:
        """Get quantil for lower control limit according to given 
        tolerance. If the data is subject to normal distribution and the 
        tolerance is given as 6, this value corresponds to the 0.135 % 
        quantile (6 sigma ~ 99.73 % of the data)."""
        if self._q_low is None:
            self._q_low = 1 - self.q_upp
        return self._q_upp

    @property
    def q_upp(self) -> float:
        """Get quantil for upper control limit according to given 
        tolerance. If the data is subject to normal distribution and the 
        tolerance is given as 6, this value corresponds to the Q_0.99865
        (0.99865-quantile or 99.865-percentile)."""
        if self._q_upp is None:
            if isinstance(self.tolerance, int):
                self._q_upp = 1 - stats.norm.cdf(self.tolerance / 2)
            else:
                self._q_upp = (1 + self.tolerance)/2
        return self._q_upp
    
    @property
    def lcl(self):
        """Get lower control limit according to given strategy and 
        tolerance."""
        if self._lcl is None:
            self._lcl, self._ucl = self._calculate_control_limits_()
        return self._lcl
    
    @property
    def ucl(self):
        """Get upper control limit according to given strategy and 
        tolerance."""
        if self._ucl is None:
            self._lcl, self._ucl = self._calculate_control_limits_()
        return self._ucl

    @property
    def limits(self) -> Tuple[float | None]:
        """Get lower and upper specification limits."""
        return (self.lsl, self.usl)

    @property
    def control_limits(self) -> Tuple[float | None]:
        """Get lower and upper control limits."""
        return (self.lcl, self.ucl)

    @property
    def tolerance(self) -> int | float:
        """Get the multiplier of the sigma tolerance for Cp and Cpk 
        value (default 6). By setting this value the cp and cpk values
        are resetted to None.
        
        If setting tolerance by giving the percentile, the percentage 
        point function is used to calculate back. The tolerance is then 
        rounded to the 4th decimal place, so that 0.9973 results in a 
        tolerance of 6 (6*sigma ~ 99.73%)."""
        return self._tolerance
    @tolerance.setter
    def tolerance(self, tolerance: int | float):
        if isinstance(tolerance, float):
            tolerance = round(stats.norm.ppf((1 + tolerance)/2), 4) * 2
        self._tolerance = tolerance
        self._reset_capabilty_values_()
    
    @property
    def cp(self) -> float | None:
        """Cp is a measure of process capability. Cp is the ratio of the 
        specification width (usl - lsl) to the process variation 
        (cp_tol*sigma). The location is not taken into account by the 
        Cp value. This value therefore only indicates the potential for 
        the Cpk value.
        This value cannot be calculated unless an upper and lower 
        specification limit is given. In this case, None is returned."""
        if None in self.limits: return None
        if self._cp is None:
            self._cp = np.abs(np.diff(self.limits))/(self.cp_tol*self.std)
        return self._cp
    
    @property
    def cpl(self) -> float:
        """Cpl is a measure of process capability. It is the ratio 
        of the distance between the process mean and the lower 
        specification limit and the lower spread of the process.
        Returns inf if no lower specification limit is specified."""
        pos = float('inf') if self.lsl is None else self.mean - self.lsl
        width = self.mean - self.q_low
        return pos/width
    
    @property
    def cpu(self) -> float:
        """Cpu is a measure of process capability. It is the ratio 
        of the distance between the process mean and the upper 
        specification limit and the upper spread of the process.
        Returns inf if no upper specification limit is specified."""
        pos = float('inf') if self.usl is None else self.usl - self.mean
        width = self.q_upp - self.mean
        return pos/width
    
    @property
    def cpk(self) -> float | None:
        """Estimates what the process is capable of producing, 
        considering  that the process mean may not be centered between 
        the specification limits. It's calculated as the minimum of
        Cpl and Cpu.
        In general, higher Cpk values indicate a more capable process. 
        Lower Cpk values indicate that the process may need improvement."""
        if self.limits == (None, None): return None
        pos_low = float('inf') if self.lsl is None else self.mean - self.lsl
        pos_upp = float('inf') if self.usl is None else self.usl - self.mean
        width_low = self.mean - self.q_low
        width_upp = self.q_upp - self.mean
        cpl = pos_low/width_low
        cpu = pos_upp/width_upp
        return min([cpl, cpu])
    
    def _calculate_control_limits_(self, strategy):
        if strategy == 'eval':
            pass # TODO implement control limits eval strategy
        elif strategy == 'fit':
            self.distribution()
            if self.dist.name == 'norm':
                lcl, ucl = self._calculate_control_limits_('norm')
            else:
                lcl = self.dist.ppf(self.q_low, **self.dist_params)
                ucl = self.dist.ppf(self.q_upp, **self.dist_params)
        elif strategy == 'norm':
            k = self.tolerance/2
            lcl = self.mean - k*self.std
            ucl = self.mean + k*self.std
        elif strategy == 'data':
            lcl = np.quantile(self.filtered, self.q_low)
            ucl = np.quantile(self.filtered, self.q_upp)
        return lcl, ucl

    def _reset_capabilty_values_(self):
        """Set all values relevant to process capability to None. This 
        function is called when one of the values relevant to the 
        calculation of capability values is adjusted (specification 
        limits or tolerance for the control limits). This ensures that 
        the process capability values are recalculated."""
        self._n_ok = None
        self._n_nok = None
        self._n_errors = None
        self._lcl = None
        self._ucl = None
        self._cp = None
        self._cpk = None
        self._q_low = None
        self._q_upp = None


def estimate_distribution(
        data: ArrayLike,
        dists: Tuple[str|rv_continuous] = DISTRIBUTION.COMMON
        ) -> Tuple[rv_continuous, float, Tuple[float]]:
    """First, the p-score is calculated by performing a 
    Kolmogorov-Smirnov test to determine how well each distribution fits
    the data. Whatever has the highest P-score is considered the most
    accurate. This is because a higher p-score means the hypothesis is
    closest to reality.
    
    Parameters
    ----------
    data : array like
        1d array of data for which a distribution is to be searched
    dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default DISTRIBUTION.COMMON
    
    Returns
    -------
    dist : scipy.stats rv_continuous
        A generic continous distribution class of best fit
    p : float
        The two-tailed p-value for the best fit
    params : tuple of floats
        Estimates for any shape parameters (if applicable), followed 
        by those for location and scale. For most random variables, 
        shape statistics will be returned, but there are exceptions 
        (e.g. norm). Can be used to generate values with the help of
        returned dist
    """
    dists = (dists, ) if isinstance(dists, (str, rv_continuous)) else dists
    results = {d: kolmogorov_smirnov_test(data, d) for d in dists}
    dist, (p, _, params) = max(results.items(), key=lambda i: i[1][0])
    if isinstance(dist, str): dist = getattr(stats, dist)
    return dist, p, params

def estimate_kernel_density(
        data: ArrayLike, scale: float | None = None, n_points: int = KDE.POINTS
    ) -> Tuple[ArrayLike, ArrayLike]:
    """Estimates the kernel density of data and returns values that are 
    useful for a plot. If those values are plotted in combination with 
    a histogram, set scale as max value of the hostogram.
    
    Kernel density estimation is a way to estimate the probability 
    density function (PDF) of a random variable in a non-parametric way. 
    The used gaussian_kde function of scipy.stats works for both 
    uni-variate and multi-variate data. It includes automatic bandwidth 
    determination. The estimation works best for a unimodal 
    distribution; bimodal or multi-modal distributions tend to be 
    oversmoothed.
    
    Parameters
    ----------
    data : array_like
        1-D array of datapoints to estimate from.
    scale : float or None, optional
        If the KDE curve is plotted in combination with other data 
        (e.g. a histogram), you can use scale to specify the height at 
        the maximum point of the KDE curve. If this value is specified, 
        the area under the curve will not be normalized, by default None
    n_points : int, optional
        Number of points the estimation and sequence should have,
        by default KDE_POINTS (defined in constants.py)

    Returns
    -------
    sequence : 1D array
        Data points at regular intervals from input data minimum to 
        maximum
    estimation : 1D array
        Data points of kernel density estimation
    """
    data = np.array(data)[~np.isnan(data)]
    sequence = np.linspace(data.min(), data.max(), n_points)
    estimation = stats.gaussian_kde(data, bw_method='scott')(sequence)
    if scale is not None:
        estimation = estimation*scale/estimation.max()
    return sequence, estimation

__all__ = [
    Estimator.__name__,
    ProcessEstimator.__name__,
    estimate_distribution.__name__,
    estimate_kernel_density.__name__,]
