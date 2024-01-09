# source for ci: https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#conf_int_of_var

import numpy as np
import pandas as pd

from typing import List
from typing import Tuple
from typing import Optional
from numpy.typing import ArrayLike
from scipy.stats._distn_infrastructure import rv_continuous

from scipy import stats
from scipy.stats import sem
from scipy.stats import skew
from scipy.stats import kurtosis

from .hypothesis import skew_test
from .hypothesis import kurtosis_test
from .hypothesis import anderson_darling_test
from .hypothesis import kolmogorov_smirnov_test
from .constants import DISTRIBUTION


class Estimator:

    __slots__ = (
        '_data', '_filtered', '_n_samples', '_mean', '_median', '_std',
        '_distribution')
    _data: pd.Series
    _filtered: pd.Series
    _n_samples: int | None
    _mean: float | None
    _median: float | None
    _std: float | None

    def __init__(self, data: ArrayLike) -> None:
        self._n_samples = 0
        self._mean = None
        self._median = None
        self._std = None
        self._data = data if isinstance(data, pd.Series) else pd.Series(data)
        self._filtered = pd.Series()
        
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
    
    def excess(self) -> Tuple[float]:
        """The curvature of the distribution corresponds to the 
        curvature of a normal distribution when the excess is close to 
        zero. Distributions with negative excess kurtosis are said to be 
        platykurtic, this distribution produces fewer and/or less 
        extreme outliers than the normal distribution (e.g. the uniform 
        distribution has no outliers). Distributions with a positive 
        excess kurtosis are said to be leptokurtic (e.g. the Laplace 
        distribution, which has tails that asymptotically approach zero 
        more slowly than a Gaussian, and therefore produces more 
        outliers than the normal distribution).

        Returns
        -------
        gamma : float
            Calculated excess value
            - gamma < 0: less extreme outliers than normal distribution
            - gamma > 0: more extreme outliers than normal distribution
        p : float
            Probability whether the excess is different from the normal 
            distribution.
        """
        gamma = kurtosis(self.filtered, fisher=True)
        p = kurtosis_test(self.filtered)
        return gamma, p
    
    def skew(self) -> Tuple[float]:
        """Compute the sample skewness of the filtered data. For 
        normally distributed data, the skewness should be about zero. 
        For unimodal continuous distributions, a skewness value greater 
        than zero means that there is more weight in the right tail of 
        the distribution.
        
        Returns
        -------
        gamma_m : float
            Calculated skew value
            - gamma_m < 0: left-skewed -> long tail left
            - gamma_m > 0: right-skewed -> long tail right
        p : float
            Probability whether the skew is different from the normal 
            distribution."""
        gamma_m = skew(self.filtered)
        p = skew_test(self.filtered)
        return gamma_m, p
    
    def norm_test_statistics(self) -> pd.DataFrame:
        pass


class ProcessEstimator(Estimator):

    __slots__ = (
        'lsl', 'usl', '_n_ok', '_n_nok', '_error_values', '_n_errors')
    _lsl: float | None
    _usl: float | None
    _n_ok: int | None
    _n_nok: int | None
    _error_values: List[float]
    _n_errrors: int

    def __init__(
            self, data: ArrayLike, lsl: Optional[float] = None, 
            usl: Optional[float] = None, error_values: List[float] = []
            ) -> None:
        self._error_values = error_values
        self._n_ok = 0
        self._n_nok = 0
        self._n_errors = 0
        self._lsl = lsl
        self._usl = usl
        super().__init__(data)

    @property
    def limits(self) -> Tuple[float | None]:
        return (self.lsl, self.usl)
        
    @property
    def filtered(self) -> pd.Series:
        """Get the data without error values and no missing value"""
        if self._filtered.empty:
            self._filtered = self.data[
                ~self.data.isin(self._error_values)
                & self.data.notna()]
        return self._filtered
    
    @property
    def n_ok(self) -> pd.Series:
        return self.filtered
    
    def process_capability(self) -> Tuple[float | None]:
        """Calculate the process capability indices Cp and CpK

        Returns
        -------
        cp : float
            estimates what the process is capable of producing if the 
            process mean were to be centered between the specification 
            limits. Assumes process output is approximately normally 
            distributed.
        cpk : float
            estimates what the process is capable of producing, considering 
            that the process mean may not be centered between the 
            specification limits.
        """
        cp = None
        cpk = None
        divisor = 3 * self.std
        if self.lsl is not None and self.usl is not None:
            cp = (self.usl - self.lsl)/(2 * divisor)
            cpk = min(self.mean - self.lsl, self.usl - self.mean) / divisor
        if self.lsl:
            cpk = (self.mean - self.lsl) / divisor
        elif self.usl:
            cpk = (self.usl - self.mean) / divisor
        return cp, cpk
    
    def distribution(
            self,
            dists: Tuple[str|rv_continuous] = DISTRIBUTION.COMMON
            ) -> Tuple[rv_continuous, float, Tuple[float]]:
        """First, the p-score is calculated by performing a 
        Kolmogorov-Smirnov test to determine how well each distribution fits
        the data. Whatever has the highest P-score is considered the most
        accurate. This is because a higher p-score means the hypothesis is
        closest to reality.
        
        Parameters
        ----------
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
        results = {d: kolmogorov_smirnov_test(self.filtered, d) for d in dists}
        dist, (p, _, params) = max(results.items(), key=lambda i: i[1][0])
        if isinstance(dist, str): dist = getattr(stats, dist)
        return dist, p, params




__all__ = [
    Estimator.__name__,
    ProcessEstimator.__name__]
