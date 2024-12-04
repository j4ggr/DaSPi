# source for ci: https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#conf_int_of_var

import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Any
from typing import Dict
from typing import Self
from typing import Tuple
from typing import Literal
from typing import Callable
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.stats._distn_infrastructure import rv_continuous

from scipy import stats
from scipy.stats import sem
from scipy.stats import skew
from scipy.stats import kurtosis

from .._typing import SpecLimit
from .._typing import SpecLimits
from .._typing import NumericSample1D

from ..constants import DIST
from ..constants import DEFAULT
from ..constants import PLOTTER
from ..constants import SIGMA_DIFFERENCE

from .confidence import cp_ci
from .confidence import cpk_ci
from .confidence import mean_ci
from .confidence import stdev_ci
from .confidence import median_ci
from .confidence import confidence_to_alpha

from .hypothesis import skew_test
from .hypothesis import kurtosis_test
from .hypothesis import ensure_generic
from .hypothesis import mean_stability_test
from .hypothesis import anderson_darling_test
from .hypothesis import variance_stability_test
from .hypothesis import kolmogorov_smirnov_test


class Estimator:

    __slots__ = (
        '_samples', '_filtered', '_n_samples', '_n_missing', '_mean', '_median', 
        '_std', '_sem', '_lcl', '_ucl', '_strategy', '_agreement', '_excess', 
        '_p_excess', '_skew', '_p_skew', '_dist', '_p_dist', '_p_ad',
        '_dist_params', 'possible_dists', '_k', '_evaluate', '_q_low', '_q_upp')
    _samples: Series
    _filtered: Series
    _n_samples: int
    _n_missing: int
    _mean: float | None
    _median: float | None
    _std: float | None
    _sem: float | None
    _lcl: float | None
    _ucl: float | None
    _excess: float | None
    _p_excess: float | None
    _skew: float | None
    _p_skew: float | None
    _dist: rv_continuous | None
    _p_dist: float | None
    _p_ad: float | None
    _dist_params: tuple | None
    _strategy: Literal['eval', 'fit', 'norm', 'data'] 
    _agreement: int | float
    possible_dists: Tuple[str | rv_continuous, ...]
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
            by default 'norm'
        agreement : int or float, optional
            Specify the tolerated process variation for which the 
            control limits are to be calculated. 
            - If int, the spread is determined using the normal 
            distribution agreement*sigma, 
            e.g. agreement = 6 -> 6*sigma ~ covers 99.75 % of the data. 
            The upper and lower permissible quantiles are then 
            calculated from this.
            - If float, the value must be between 0 and 1.This value is
            then interpreted as the acceptable proportion for the 
            spread, e.g. 0.9973 (which corresponds to ~ 6 sigma)
            by default 6
        possible_dists : tuple of strings or rv_continous, optional
            Distributions to which the data may be subject. Only 
            continuous distributions of scipy.stats are allowed,
            by default `DIST.COMMON`
        evaluate : callable or None, optional
            Function that takes this instance as argument and returns
            one of the allowed strategy {'eval', 'fit', 'norm', 'data'},
            by default None   
        """
        self._mean = None
        self._median = None
        self._std = None
        self._sem = None
        self._lcl = None
        self._ucl = None
        self._q_low = None
        self._q_upp = None
        self._excess = None
        self._p_excess = None
        self._skew = None
        self._p_skew = None
        self._dist = None
        self._p_dist = None
        self._p_ad = None
        self._dist_params = None
        self.possible_dists = possible_dists
        self._filtered = pd.Series()
        if not isinstance(samples, pd.Series):
            samples = pd.Series(samples)
        self._samples = samples
        self._n_samples = len(self.samples)
        self._n_missing = self.samples.isna().sum()
        self._strategy = 'norm'
        self.strategy = strategy
        self._agreement = -1
        self.agreement = agreement
        self._evaluate = evaluate
    
    @property
    def _descriptive_statistic_attrs_(self) -> Tuple[str, ...]:
        """Get attribute names used for `describe` method."""
        attrs = (
            'n_samples', 'n_missing', 'mean', 'median', 'std', 'sem', 'excess',
            'p_excess', 'skew', 'p_skew', 'dist', 'p_ad', 'lcl', 'ucl',
            'strategy')
        return attrs

    @property
    def samples(self) -> pd.Series:
        """Get the raw samples as it was given during instantiation
        as pandas Series."""
        return self._samples
        
    @property
    def filtered(self) -> pd.Series:
        """Get the samples without error values and no missing value"""
        if self._filtered.empty:
            self._filtered = pd.to_numeric(self.samples[self.samples.notna()])
        return self._filtered
    
    @property
    def n_samples(self) -> int:
        """Get sample size of unfiltered samples"""
        return self._n_samples
    
    @property
    def n_missing(self) -> int:
        """Get amount of missing values"""
        return self._n_missing
    
    @property
    def dof(self) -> int:
        """Get degree of freedom for filtered samples"""
        return len(self.filtered)-1

    @property
    def mean(self) -> float:
        """Get mean of filtered samples"""
        if self._mean is None:
            self._mean = self.filtered.mean()
        return self._mean

    @property
    def median(self) -> float:
        """Get median of filtered samples"""
        if self._median is None:
            self._median = self.filtered.median()
        return self._median
    
    @property
    def std(self) -> float:
        """Get standard deviation of filtered samples"""
        if self._std is None:
            self._std = self.filtered.std()
        return self._std
    
    @property
    def sem(self) -> float:
        """Get standard error mean of filtered samples"""
        if self._sem is None:
            self._sem = float(self.filtered.sem()) # type: ignore
        return self._sem
    
    @property
    def lcl(self) -> float:
        """Get lower control limit according to given strategy and 
        agreement."""
        if self._lcl is None:
            self._lcl, self._ucl = self._calculate_control_limits_()
        return self._lcl
    
    @property
    def ucl(self) -> float:
        """Get upper control limit according to given strategy and 
        agreement."""
        if self._ucl is None:
            self._lcl, self._ucl = self._calculate_control_limits_()
        return self._ucl

    @property
    def q_low(self) -> float:
        """Get quantil for lower control limit according to given 
        agreement. If the samples is subject to normal distribution and 
        the agreement is given as 6, this value corresponds to the 0.135 % 
        quantile (6 sigma ~ 99.73 % of the samples)."""
        if self._q_low is None:
            self._q_low = float(stats.norm.cdf(-self.agreement/2))
        return self._q_low

    @property
    def q_upp(self) -> float:
        """Get quantil for upper control limit according to given 
        agreement. If the sample data is subject to normal distribution 
        and the agreement is given as 6, this value corresponds to the 
        Q_0.99865 (0.99865-quantile or 99.865-percentile)."""
        if self._q_upp is None:
            self._q_upp = 1 - self.q_low
        return self._q_upp
    
    @property
    def excess(self) -> float:
        """Get the Fisher kurtosis (excess) of filtered samples.
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
            self._excess = float(kurtosis(
                self.filtered, fisher=True, bias=False))
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
    def skew(self) -> float:
        """Get the skewness of the filtered samples.
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
        normal distribution"""
        if self._p_skew is None:
            self._p_skew = skew_test(self.filtered)[0]
        return self._p_skew
    
    @property
    def p_ad(self) -> float:
        """Get the probability that the filtered samples are subject of
        the normal distribution by performing a Anderson-Darling test."""
        if self._p_ad is None:
            self._p_ad = anderson_darling_test(self.filtered)[0]
        return self._p_ad

    @property
    def dist(self) -> rv_continuous:
        """Get fitted distribution. None if method distribution has 
        not been called."""
        if self._dist is None:
            self._dist, self._p_dist, self._dist_params = self.distribution()
        return self._dist
    
    @property
    def p_dist(self) -> float:
        """Get probability of fitted distribution. None if method 
        distribution has not been called."""
        if self._p_dist is None:
            self._dist, self._p_dist, self._dist_params = self.distribution()
        return self._p_dist
    
    @property
    def dist_params(self) -> Tuple[float, ...]:
        """Get params of fitted distribution. None if method 
        distribution has not been called."""
        if self._dist_params is None:
            self._dist, self._p_dist, self._dist_params = self.distribution()
        return self._dist_params
    
    @property
    def strategy(self) -> Literal['eval', 'fit', 'norm', 'data']:
        """Strategy used to determine the control limits (can also be 
        interpreted as the process range).

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
        """Get the multiplier of the sigma agreement for Cp and Cpk 
        value (default 6). By setting this value the cp and cpk values
        are resetted to None.
        
        If setting agreement by giving the percentile, enter the 
        acceptable proportion for the spread, e.g. 0.9973 
        (which corresponds to ~ 6 sigma)"""
        return self._agreement
    @agreement.setter
    def agreement(self, agreement: int | float) -> None:
        assert agreement > 0, (
            'Agreement must be set as a percentage (0 < agreement < 1) '
            + 'or as a multiple of the standard deviation (agreement >= 1), '
            + f'got {agreement}.')
        
        if agreement >= 1:
            self._k = agreement / 2
        else:
            self._k = float(stats.norm.ppf((1 + agreement) / 2))
            agreement = 2 * self._k
        
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
        return estimate_distribution(self.filtered, self.possible_dists)
    
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
                lcl = float(self.dist.ppf(self.q_low, *self.dist_params))
                ucl = float(self.dist.ppf(self.q_upp, *self.dist_params))
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
    
    def describe(self) -> Series:
        """Generate descriptive statistics.
        
        Returns
        -------
        stats : Series
            Summary statistics as pandas Series 
        """
        def _value_(a: str) -> float | int | str:
            return getattr(self, a).name if a == 'dist' else getattr(self, a)
        stats = pd.Series(
            {a: _value_(a) for a in self._descriptive_statistic_attrs_})
        return stats


class ProcessEstimator(Estimator):

    __slots__ = (
        '_lsl', '_usl', '_n_ok', '_n_nok', '_error_values', '_n_errors', 
        '_cp', '_cpk', '_Z', '_Z_lt')
    _lsl: SpecLimit
    _usl: SpecLimit
    _n_ok: int | None
    _n_nok: int | None
    _error_values: Tuple[float, ...]
    _n_errrors: int
    _cp: float | None
    _cpk: float | None
    _Z: float | None
    _Z_lt: float | None

    def __init__(
            self,
            samples: NumericSample1D, 
            lsl: SpecLimit = None, 
            usl: SpecLimit = None, 
            error_values: Tuple[float, ...] = (),
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: float | int = 6, 
            possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON
            ) -> None:
        """"An object for various statistical estimators. This class 
        extends the estimator with process-specific statistics such as 
        specification limits, Cp and Cpk values
        
        The attributes are calculated lazily. After the class is 
        instantiated, all attributes are set to None. As soon as an 
        attribute (actually Property) is called, the value is calculated
        and stored so that the calculation is only performed once

        Parameters
        ----------
        samples : NumericSample1D
            1D array of process data.
        lsl, usl : float
            Lower and upper specification limit for process data.
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
            distribution agreement*sigma, 
            e.g. agreement = 6 -> 6*sigma ~ covers 99.75 % of the data. 
            The upper and lower permissible quantiles are then 
            calculated from this.
            - If float, the value must be between 0 and 1.This value is
            then interpreted as the acceptable proportion for the 
            spread, e.g. 0.9973 (which corresponds to ~ 6 sigma)
            by default 6
        possible_dists : tuple of strings or rv_continous, optional
            Distributions to which the data may be subject. Only 
            continuous distributions of scipy.stats are allowed,
            by default DIST.COMMON
        """
        self._n_ok = None
        self._n_nok = None
        self._n_errors = None
        self._cp = None
        self._cpk = None
        self._Z = None
        self._Z_lt = None
        self._error_values = error_values
        self._lsl = None
        self._usl = None
        self.lsl = lsl
        self.usl = usl
        self._reset_values_()
        super().__init__(samples, strategy, agreement, possible_dists)
    
    @property
    def _descriptive_statistic_attrs_(self) -> Tuple[str, ...]:
        """Get attribute names used for `describe` method (read-only)."""
        _attrs = super()._descriptive_statistic_attrs_
        attrs = (_attrs[:2]
                 + ('n_ok', 'n_nok', 'n_errors')
                 + _attrs[2:]
                 + ('lsl', 'usl', 'cp', 'cpk', 'Z', 'Z_lt'))
        return attrs
        
    @property
    def filtered(self) -> Series:
        """Get the data without error values and no missing value
         (read-only)."""
        if self._filtered.empty:
            self._filtered = pd.to_numeric(
                self.samples[
                    ~ self.samples.isin(self._error_values)
                    & self.samples.notna()])
        return self._filtered
    
    @property
    def n_ok(self) -> int:
        """Get amount of OK-values (read-only)."""
        if self._n_ok is None:
            self._n_ok = (self.n_samples
                          - self.n_nok
                          - self.n_errors
                          - self.n_missing)
        return self._n_ok
    
    @property
    def n_nok(self) -> int:
        """Get amount of NOK-values (read-only)."""
        if self._n_nok is None:
            self._n_nok = (
                (self.filtered >= self.usl).sum() if self.usl else 0
                + (self.filtered <= self.lsl).sum() if self.lsl else 0)
        return self._n_nok
    
    @property
    def p_ok(self) -> float:
        """Get amount of OK-values as percent (read-only)."""
        return 100*self.n_ok/self.n_samples
    
    @property
    def p_nok(self) -> float:
        """Get amount of NOK-values as percent (read-only)."""
        return 100*self.n_nok/self.n_samples
    
    @property
    def n_errors(self) -> int:
        """Get amount of error values (read-only)."""
        if self._n_errors is None:
            self._n_errors = self.samples.isin(self._error_values).sum()
        return self._n_errors

    @property
    def lsl(self) -> SpecLimit:
        """Get and set lower specification limit."""
        if self._lsl is not None and self._usl is not None:
            assert self._lsl < self._usl
        return self._lsl
    @lsl.setter
    def lsl(self, lsl: SpecLimit) -> None:
        if self._lsl != lsl:
            self._lsl = lsl if pd.notna(lsl) else None
            self._reset_values_()

    @property
    def usl(self) -> SpecLimit:
        """Get and set upper specification limit."""
        if self._lsl is not None and self._usl is not None:
            assert self._usl > self._lsl
        return self._usl
    @usl.setter
    def usl(self, usl: SpecLimit) -> None:
        if self._usl != usl:
            self._usl = usl if pd.notna(usl) else None
            self._reset_values_()

    @property
    def limits(self) -> SpecLimits:
        """Get lower and upper specification limits (read-only)."""
        return (self.lsl, self.usl)

    @property
    def control_limits(self) -> Tuple[float, float]:
        """Get lower and upper control limits (read-only)."""
        return (self.lcl, self.ucl)
    
    @property
    def tolerance_range(self) -> float:
        """Get tolerance range. If one of the specification limits is 
        not specified, inf is returned (read-only)."""
        if self.usl is None or self.lsl is None:
            return float('inf')
        return self.usl - self.lsl
    
    @property
    def cp(self) -> float | None:
        """Cp is a measure of process capability. Cp is the ratio of the 
        specification width (usl - lsl) to the process variation 
        (agreement*sigma). The location is not taken into account by the 
        Cp value. This value therefore only indicates the potential for 
        the Cpk value.
        This value cannot be calculated unless an upper and lower 
        specification limit is given. In this case, None is returned."""
        if None in self.limits:
            return None
        
        if self._cp is None:
            self._cp = self.tolerance_range / (self.agreement * self.std)
        return self._cp
    
    @property
    def cpl(self) -> float:
        """Cpl is a measure of process capability. It is the ratio 
        of the distance between the process mean and the lower 
        specification limit and the lower spread of the process.
        Returns inf if no lower specification limit is specified."""
        if self.lsl is None:
            return float('inf')
        return (self.mean - self.lsl) / (self.mean - self.lcl)
    
    @property
    def cpu(self) -> float:
        """Cpu is a measure of process capability. It is the ratio 
        of the distance between the process mean and the upper 
        specification limit and the upper spread of the process.
        Returns inf if no upper specification limit is specified."""
        if self.usl is None:
            return float('inf')
        return (self.usl - self.mean) / (self.ucl - self.mean)
    
    @property
    def cpk(self) -> float | None:
        """Estimates what the process is capable of producing, 
        considering  that the process mean may not be centered between 
        the specification limits. It's calculated as the minimum of
        Cpl and Cpu.
        In general, higher Cpk values indicate a more capable process. 
        Lower Cpk values indicate that the process may need improvement."""
        if self.limits == (None, None):
            return None
        return min([self.cpl, self.cpu])
    
    @property
    def Z(self) -> float:
        """The Sigma level Z is another process capability indicator 
        alongside cp and cpk. It describes how many standard deviations 
        can be placed between the mean value and the nearest tolerance 
        limit of a process."""
        if self._Z is None:
            limit: float = self.lsl if self.cpl < self.cpu else self.usl # type: ignore
            self._Z = abs(self.z_transform(limit))
        return self._Z
    
    @property
    def Z_lt(self) -> float:
        """Statements about long-term capabilities can be derived from 
        short-term capabilities using the sigma level. The empirically 
        determined value of 1.5 is subtracted from the sigma level."""
        if self._Z_lt is None:
            self._Z_lt = self.Z - SIGMA_DIFFERENCE
        return self._Z_lt

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
        self._cp = None
        self._cpk = None
        self._Z = None
        self._Z_lt = None

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

def estimate_kernel_density(
        data: NumericSample1D,
        stretch: float = 1,
        height: float | None = None, 
        base: float = 0,
        n_points: int = DEFAULT.KD_SEQUENCE_LEN
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
    sequence = np.linspace(data.min(), data.max(), n_points)
    estimation = stats.gaussian_kde(data, bw_method='scott')(sequence)
    stretch = stretch if height is None else height/estimation.max()
    estimation = stretch*estimation + base
    return sequence, estimation

def estimate_kernel_density_2d(
        feature: NumericSample1D,
        target: NumericSample1D,
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
    f_min = feature.min()
    f_max = feature.max()
    f_margin = (f_max - f_min) * margin
    t_min = target.min()
    t_max = target.max()
    t_margin = (t_max - t_min) * margin
    feature_seq, target_seq = np.meshgrid(
        np.linspace(f_min - f_margin, f_max + f_margin, n_points),
        np.linspace(t_min - t_margin, t_max + t_margin, n_points))
    _values = np.vstack([feature, target])
    _sequences = np.vstack([feature_seq.ravel(), target_seq.ravel()])
    estimation = stats.gaussian_kde(_values, bw_method='scott')(_sequences)
    estimation = np.reshape(estimation.T, feature_seq.shape)
    return feature_seq, target_seq, estimation

def estimate_capability_confidence(
        samples: NumericSample1D, 
        lsl: SpecLimit, 
        usl: SpecLimit,
        kind: Literal['cp', 'cpk'] = 'cpk',
        level: float = 0.95,
        n_groups: int = 1,
        **kwds
        ) -> Tuple[float, float, float]:
    """Calculates the confidence interval for the process capability 
    index (Cp or Cpk) of a process.
    
    This function is an extension of the `cp_ci` and `cpk_ci` functions.
    It instantiates a `ProcessEstimator` and then determines the 
    confidence intervals using the Cp or Cpk values from the estimator.
    
    Parameters
    ----------
    samples : NumericSample1D
        1D array of process data.
    lsl, usl : float
        Lower and upper specification limit for process data.
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
    estimator = ProcessEstimator(samples, lsl, usl, **kwds)
    assert kind in ('cp', 'cpk'), f'Unkown value for {kind=}'
    
    if kind == 'cp':
        if estimator.cp is None:
            raise ValueError(
                'To calculate the cp values, both limits must be provided')
        ci_values = cp_ci(
            cp=estimator.cp,
            n_samples=estimator.n_samples,
            level=level,
            n_groups=n_groups)
    
    elif kind == 'cpk':
        if estimator.cpk is None:
            raise ValueError(
                'At least one spec limit must be provided')
        ci_values = cpk_ci(
            cpk=estimator.cpk,
            n_samples=estimator.n_samples,
            level=level,
            n_groups=n_groups)

    return ci_values


class Lowess:
    """Smooth the data using Locally Estimated Weighted Scatterplot 
    Smoothing (LOWESS).
    
    LOWESS is not necessarily suitable for all regression models due to 
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
    
    
    Notes
    -----
    This class was created using the following Medium platform article:

    Soul Dobilas (2020), LOWESS Regression in Python: How to Discover
    Clear Patterns in Your Data?, 
    [Towards Data Science](https://towardsdatascience.com/lowess-regression-in-python-how-to-discover-clear-patterns-in-your-data-f26e523d7a35)
    """
    __slots__ = (
        'source', 'target', 'feature', '_lowess_output',
        )
    
    source: DataFrame
    """The data source for the plot"""
    target: str
    """The column name of the target variable."""
    feature: str
    """The column name of the feature variable."""
    _lowess_output: NDArray
    """Smoothed raw data coming from `statsmodels.nonparametric.lowess`
    function"""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
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
    
    @property
    def fitted(self) -> bool:
        """True if the data has been fitted (read-only)."""
        return hasattr(self, '_lowess_output')

    @property
    def n_samples(self) -> int:
        """Amount of samples in source after removing missing values
        (read-only)."""
        return len(self.source)
    
    @property
    def lowess_target(self) -> NDArray:
        """Lowess output target values (read-only)."""
        self._check_fitted_('lowess_target')
        return self._lowess_output[:, 1]
    
    @property
    def lowess_feature(self) -> NDArray:
        """Lowess output feature values (read-only)."""
        self._check_fitted_('lowess_feature')
        return self._lowess_output[:, 0]
    
    @property
    def residuals(self) -> NDArray:
        """Residuals as difference between target and lowess (read-only)."""
        self._check_fitted_('residuals')
        return self.source[self.target].to_numpy() - self.lowess_target
    
    @staticmethod
    def prob_points(
            n_points: int,
            fraction: float | None = None
            ) -> NDArray:
        """Ordinates For Probability Plotting. Numpy analogue of `R`'s 
        `ppoints` function.

        The prob_points function generates a sequence of points based on 
        the empirical distribution of the data.

        Parameters
        ----------
        n : int
            Number of points generated
        fraction : float | None, optional
            Offset fraction (typically between 0 and 1). If None, the 
            offset is set to 3/8 if n <= 10 and 2/3 otherwise,
            by default None

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
        assert isinstance(n_points, int) and n_points > 0, (
            f'Number of points must be a positive integer, got {n_points=}')
        if fraction is None:
            fraction = 3/8 if n_points <= 10 else 2/3
        assert fraction < n_points/2, (
            f'Offset {fraction=} must be less than half the number of points')
        ppoints = (np.arange(n_points) + 1 - fraction) / (n_points + 1 - 2*fraction)
        return ppoints
    
    def _check_fitted_(self, method_name: str) -> None:
        """Check if the model has been fitted.
        
        Raises
        ------
        AssertionError
            If the model has not been fitted yet using the fit method.
        """
        assert self.fitted, (
            f'The data must be fitted before {method_name} can be performed.')

    def fit(self, fraction: float) -> Self:
        """Fits the model using the `statsmodels.nonparametric.lowess` 
        method.
        
        Parameters
        ----------
        fraction : float
            The fraction of the data used for each local regression. A 
            good value to start with is 2/3 (default value of 
            statsmodels). Reduce the value to avoid underfitting. 
            A value below 0.2 usually leads to overfitting.
        
        Returns
        -------
        Lowess:
            The instance of the Lowess with the fitted values.
        """
        
        self._lowess_output = sm.nonparametric.lowess(
            endog=self.source[self.target],
            exog=self.source[self.feature],
            frac=fraction,
            is_sorted=True,
            return_sorted=True)
        return self

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
            x=self.lowess_feature,
            y=self.lowess_target,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True
            ) | kwds
        return interp1d(**_kwds)
    
    def predict(
        self,
        feature: int | float | NumericSample1D,
        kind: str | int = 'linear') -> NDArray:
        """Predict the target value(s) for the given feature value(s).

        Parameters
        ----------
        feature : int | float | NumericSample1D
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
        if isinstance(feature, (int, float)):
            feature = [feature]
        return self._interpolate_(kind=kind)(feature)
        
    def smoothed_ci(
            self,
            n_points: int = DEFAULT.LOWESS_SEQUENCE_LEN,
            level: float = 0.95
            ) -> DataFrame:
        """Calculates the LOWESS smoothed line and its confidence bands.

        The function generates a sequence of feature values, predicts 
        the corresponding target values using the LOWESS model, and 
        then calculates the lower and upper confidence bands based on 
        the distribution of the residuals.
    
        Parameters
        ----------
        n_points : int, optional
            Number of points the smoothed line and its sequence should 
            have, by default LOWESS_SEQUENCE_LEN 
            (defined in constants.py)
        level : float in (0, 1), optional
            confidence level, by default 0.95

        Returns
        -------
        DataFrame:
            A pandas DataFrame containing the following columns:
            - `PLOTTER.LOWESS_FEATURE`: The sequence of feature values.
            - `PLOTTER.LOWESS_TARGET`: The predicted target values.
            - `PLOTTER.LOWESS_LOW`: The lower confidence band.
            - `PLOTTER.LOWESS_UPP`: The upper confidence band.
        """
        self._check_fitted_('smoothed_ci')
        alpha = confidence_to_alpha(level, two_sided=True)
        
        sequence = np.linspace(
            min(self.lowess_feature), max(self.lowess_feature), n_points)
        prediction = self.predict(sequence, kind='linear')
        
        smooth_residuals = self._interpolate_(
                x=self.lowess_feature, y=self.residuals)
        residuals = smooth_residuals(sequence)
        ppoints = self.prob_points(n_points)
        quantile = np.quantile(residuals, alpha * ppoints)

        data = DataFrame({
            PLOTTER.LOWESS_FEATURE: sequence,
            PLOTTER.LOWESS_TARGET: prediction,
            PLOTTER.LOWESS_LOW: prediction - quantile,
            PLOTTER.LOWESS_UPP: prediction + quantile})
        return data


__all__ = [
    'Estimator',
    'ProcessEstimator',
    'estimate_distribution',
    'estimate_kernel_density',
    'estimate_kernel_density_2d',
    'estimate_capability_confidence',
    'Lowess']
