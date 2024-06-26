# source for ci: https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#conf_int_of_var

import numpy as np
import pandas as pd

from typing import Tuple
from typing import Literal
from typing import Sequence
from typing import Optional
from typing import Callable
from numpy.typing import NDArray
from numpy.typing import ArrayLike
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
from ..constants import PLOTTER


from .confidence import mean_ci
from .confidence import stdev_ci
from .confidence import median_ci

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
            if isinstance(self.agreement, int):
                self._q_low = float(stats.norm.cdf(-self.agreement/2))
            else:
                self._q_low = (1 - self.agreement)/2
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
    def agreement(self) -> int | float:
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
        if self._agreement != agreement:
            self._agreement = agreement
            self._reset_values_()

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
        '_cp', '_cpk')
    _lsl: SpecLimit
    _usl: SpecLimit
    _n_ok: int | None
    _n_nok: int | None
    _error_values: Tuple[float, ...]
    _n_errrors: int
    _cp: int | None
    _cpk: int | None

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
        self._error_values = error_values
        self._lsl = None
        self._usl = None
        self.lsl = lsl
        self.usl = usl
        self._reset_values_()
        super().__init__(samples, strategy, agreement, possible_dists)
    
    @property
    def _descriptive_statistic_attrs_(self) -> Tuple[str, ...]:
        """Get attribute names used for `describe` method."""
        _attrs = super()._descriptive_statistic_attrs_
        attrs = (_attrs[:2]
                 + ('n_ok', 'n_nok', 'n_errors')
                 + _attrs[2:]
                 + ('lsl', 'usl', 'cp', 'cpk'))
        return attrs
        
    @property
    def filtered(self) -> Series:
        """Get the data without error values and no missing value"""
        if self._filtered.empty:
            self._filtered = pd.to_numeric(
                self.samples[
                    ~ self.samples.isin(self._error_values)
                    & self.samples.notna()])
        return self._filtered
    
    @property
    def n_ok(self) -> int:
        if self._n_ok is None:
            self._n_ok = (self.n_samples
                          - self.n_nok
                          - self.n_errors
                          - self.n_missing)
        return self._n_ok
    
    @property
    def n_nok(self) -> int:
        if self._n_nok is None:
            self._n_nok = (
                (self.filtered >= self.usl).sum() if self.usl else 0
                + (self.filtered <= self.lsl).sum() if self.lsl else 0)
        return self._n_nok
    
    @property
    def p_ok(self) -> float:
        """Get amount of OK-values as percent"""
        return 100*self.n_ok/self.n_samples
    
    @property
    def p_nok(self) -> float:
        """Get amount of NOK-values as percent"""
        return 100*self.n_nok/self.n_samples
    
    @property
    def n_errors(self) -> int:
        if self._n_errors is None:
            self._n_errors = self.samples.isin(self._error_values).sum()
        return self._n_errors

    @property
    def lsl(self) -> SpecLimit:
        """Get lower specification limit"""
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
        """Get upper specification limit"""
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
        """Get lower and upper specification limits."""
        return (self.lsl, self.usl)

    @property
    def control_limits(self) -> Tuple[float, float]:
        """Get lower and upper control limits."""
        return (self.lcl, self.ucl)
    
    @property
    def cp_tol(self) -> float:
        return 2*self._k
    
    @property
    def cp(self) -> float | None:
        """Cp is a measure of process capability. Cp is the ratio of the 
        specification width (usl - lsl) to the process variation 
        (agreement*sigma). The location is not taken into account by the 
        Cp value. This value therefore only indicates the potential for 
        the Cpk value.
        This value cannot be calculated unless an upper and lower 
        specification limit is given. In this case, None is returned."""
        if self.usl is None or self.lsl is None:
            return None
        if self._cp is None:
            agreement = 2 * self._k
            self._cp = (self.usl - self.lsl)/(agreement*self.std) # type: ignore
        return self._cp
    
    @property
    def cpl(self) -> float:
        """Cpl is a measure of process capability. It is the ratio 
        of the distance between the process mean and the lower 
        specification limit and the lower spread of the process.
        Returns inf if no lower specification limit is specified."""
        space = float('inf') if self.lsl is None else self.mean - self.lsl
        spread = self.mean - self.lcl
        return space/spread
    
    @property
    def cpu(self) -> float:
        """Cpu is a measure of process capability. It is the ratio 
        of the distance between the process mean and the upper 
        specification limit and the upper spread of the process.
        Returns inf if no upper specification limit is specified."""
        space = float('inf') if self.usl is None else self.usl - self.mean
        spread = self.ucl - self.mean
        return space/spread
    
    @property
    def cpk(self) -> float | None:
        """Estimates what the process is capable of producing, 
        considering  that the process mean may not be centered between 
        the specification limits. It's calculated as the minimum of
        Cpl and Cpu.
        In general, higher Cpk values indicate a more capable process. 
        Lower Cpk values indicate that the process may need improvement."""
        if self.limits == (None, None): return None
        return min([self.cpl, self.cpu])

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
        data: ArrayLike, stretch: float = 1, height: float | None = None, 
        base: float = 0, n_points: int = PLOTTER.KD_SEQUENCE_LEN
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
    data : array_like
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

__all__ = [
    'Estimator',
    'ProcessEstimator',
    'estimate_distribution',
    'estimate_kernel_density',]
