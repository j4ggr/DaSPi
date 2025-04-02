import random

import numpy as np

from math import pi
from math import sin
from typing import Any
from typing import Tuple
from typing import Literal
from dataclasses import dataclass
from numpy.typing import NDArray
from pandas.core.series import Series

from .._typing import FloatOrArray


@dataclass(frozen=True)
class SpecLimits:
    """Class to hold the specification limits of a parameter.

    Parameters
    ----------
    lower : float, optional
        The lower limit of the specification. Default is -inf.
    upper : float, optional
        The upper limit of the specification. Default is inf.
    """
    lower: float = float('-inf')
    upper: float = float('inf')

    def __post_init__(self) -> None:
        assert  self.lower <= self.upper, (
            'Lower limit must be less than or equal to the upper limit.')

    @property
    def is_unbounded(self) -> bool:
        """Check if any of lower or upper is -inf or inf."""
        return self.lower == float('-inf') or self.upper == float('inf')

    @property
    def are_both_finite(self) -> bool:
        """Check if both lower and upper are finite values."""
        return (self.lower > float('-inf')) and (self.upper < float('inf'))

    @property
    def both_unbounded(self) -> bool:
        """Check if both lower and upper are -inf and inf, respectively."""
        return self.lower == float('-inf') and self.upper == float('inf')
    
    @property
    def range(self) -> float:
        """Returns the difference between upper and lower limits."""
        return self.upper - self.lower
    
    def to_tuple(self) -> Tuple[float, float]:
        """Returns the lower and upper limits as a tuple.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the lower and upper limits in order.
        """
        return (self.lower, self.upper)

    def __contains__(self, value: float) -> bool:
        """Check if a given value is within the specified limits.

        Parameters
        ----------
        value : float
            The value to check against the limits.

        Returns
        -------
        bool
            True if the value is within the limits, False otherwise.
        """
        return self.lower <= value <= self.upper
    
    def __eq__(self, other: Any) -> bool:
        """
        Checks if this SpecLimits instance is equal to another 
        SpecLimits instance or a tuple.

        Parameters
        ----------
        other : SpecLimits, list or tuple
            The object to compare with this instance.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        if isinstance(other, SpecLimits):
            return self.to_tuple() == other.to_tuple()
        elif isinstance(other, (tuple, list)) and len(other) == 2:
            return self.to_tuple() == tuple(other)
        return False
    

class Parameter:
    """Class to hold the limits, tolerance, and nominal of a parameter.

    Parameters
    ----------
    limits : SpecLimits | Tuple[float, float] | None, optional
        The limits of the parameter. The limits are automatically 
        calculated if only tolerance is given. Default is None.
    tolerance : float or None, optional
        The tolerance of the parameter. Default is None.
    nominal : float or None, optional
        The nominal value of the parameter. Default is None.
    precision : int, optional
        The precision to counteract the rounding error of floating-point
        numbers. Default is 10.

    Notes
    -----
    The nominal and tolerance are automatically calculated if only 
    limits are provided. The limits and nominal are automatically 
    calculated if only tolerance is provided.
    - Tolerance is calculated as the difference between the upper and 
      lower limits.
    - Nominal is calculated as the average of the lower and upper limits.
    - Limits are calculated as the nominal +/- tolerance / 2.

    Raises
    ------
    AssertionError
        If the lower limit is greater than the upper limit.
    AssertionError
        If limits, tolerance and nominal are not provided.

    Examples
    --------
    Create a parameter with limits:
    ``` python
    Parameter(limits=(0, 1))
    ```

    ```
    Parameter(limits=(0, 1), tolerance=1.0, nominal=0.5)
    ```
    
    Create a parameter with tolerance and nominal:
    ``` python
    Parameter(tolerance=0.1, nominal=0.5)
    ```
    
    ```
    Parameter(limits=(0.45, 0.55), tolerance=0.1, nominal=0.5)
    ```
    """

    __slots__ = ('_limits', '_tolerance', '_nominal')

    _limits: SpecLimits
    """The limits of the parameter."""

    _tolerance: float
    """The tolerance of the parameter."""

    _nominal: float
    """The nominal value of the parameter."""

    def __init__(
            self,
            *,
            limits: SpecLimits | Tuple[float, float] | None = None, 
            tolerance: float | None = None,
            nominal: float | None = None,
            precision: int = 10
            ) -> None:
        if isinstance(limits, tuple):
            limits = SpecLimits(*limits)
        elif limits is None:
            assert tolerance is not None and nominal is not None, (
                'Either limits or tolerance and nominal must be provided.')
            limits = SpecLimits(
                round(nominal - tolerance / 2, precision),
                round(nominal + tolerance / 2, precision))

        if tolerance is None:
            assert limits.are_both_finite, (
                'If tolerance is not provided, limits must be finite.')
            tolerance = round(limits.upper - limits.lower, precision)
        
        if nominal is None:
            assert limits.are_both_finite, (
                'If nominal is not provided, limits must be finite.')
            nominal = round(limits.lower + tolerance / 2, precision)

        self._limits = limits
        self._tolerance = tolerance
        self._nominal = nominal
    
    @property
    def LIMITS(self) -> SpecLimits:
        """The lower and upper limits of the parameter (read-only)."""
        return self._limits
    
    @property
    def TOLERANCE(self) -> float:
        """The tolerance of the parameter."""
        return self._tolerance
    
    @property
    def NOMINAL(self) -> float:
        """The nominal value of the parameter."""
        return self._nominal
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(limits={self.LIMITS.to_tuple()}, '
            f'tolerance={self.TOLERANCE}, nominal={self.NOMINAL})')

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'limits={self.LIMITS.to_tuple()}, tolerance={self.TOLERANCE}, '
            f'nominal={self.NOMINAL})')


class RandomProcessValue:
    """Class to generate a random process value.
    
    Parameters
    ----------
    nominal : float
        The base value around which the random value will be generated.
    tolerance : float
        The tolerance range within which the generated value will be
        generated.
    dist : {'normal', 'uniform', 'circular'}
        The distribution from which the random value will be generated.
    clip_within_tolerance : bool, optional
        Whether to clip the generated value to the range defined by the
        tolerance. Default is False.
    """

    _allowed_dists: Tuple[str, ...] = ('normal', 'uniform', 'circular')
    """Allowed distributions for the RandomProcessValue class."""

    def __init__(
            self,
            nominal: float,
            tolerance: float,
            dist: Literal['normal', 'uniform', 'circular'],
            clip_within_tolerance: bool = False
            ) -> None:
        """Initialize the RandomProcessValue class."""
        assert dist in self._allowed_dists, (
            f'Distribution must be one of {self._allowed_dists}, got {dist}.')
        self.nominal = nominal
        self.tolerance = tolerance
        self.dist = dist
        self.clip_within_tolerance = clip_within_tolerance
    
    @property
    def scale(self) -> float:
        """Scale factor for the distribution (6 sigma rule)."""
        return self.tolerance / 6
    
    @property
    def lower(self) -> float:
        """Lower bound of the distribution."""
        return self.nominal - self.tolerance / 2
    
    @property
    def upper(self) -> float:
        """Upper bound of the distribution."""
        return self.nominal + self.tolerance / 2
    
    @staticmethod
    def clip(value: float, min_value: float, max_value: float) -> float:
        """Clip a value between a minimum and maximum value.
        
        This function ensures that the provided value is within the 
        specified range, clipping it if it falls outside the range.
        
        Parameters
        ----------
        value : float
            The value to be clipped.
        min_value : float
            The minimum value of the range.
        max_value : float
            The maximum value of the range.
        
        Returns
        -------
        float
            The clipped value.
        """
        return max(min(value, max_value), min_value)

    def normal(self) -> float:
        """Generate a random value within the specified tolerance range.
        
        This function generates a random value within a range defined by
        the provided nominal and tolerance. The generated value is
        calculated by a normal distribution with a mean of the nominal
        and a standard deviation equal to the tolerance divided by 6.
        
        Returns
        -------
        float
            The generated random value within the specified tolerance 
            range.
        """
        value = random.normalvariate(self.nominal, self.scale)
        if self.clip_within_tolerance:
            value = self.clip(value, self.lower, self.upper)
        return value
    
    def uniform(self) -> float:
        """Generate a random value within the specified tolerance range.
        
        This method generates a random value within the specified 
        tolerance range using a uniform distribution.
        
        Returns
        -------
        float
            The generated random value within the specified tolerance 
            range.
        """
        value = random.uniform(self.lower, self.upper)
        return value
    
    def circular(self) -> float:
        """Generate a randomized offset based on a circular distribution.

        This function computes a randomized value by applying a circular 
        offset to the provided tolerance. The offset is determined by 
        multiplying the tolerance by the sine of a random angle uniformly 
        distributed between 0 and 2π radians, simulating the effects of 
        coaxiality in all possible directions.

        Parameters
        ----------
        value : float
            The base value to which the circular offset will be applied.
        
        Returns
        -------
        float
            The modified value after applying the random circular offset.
        """
        return self.tolerance * sin(random.uniform(0, 2 * pi))
    
    def __call__(self) -> float:
        """Generate a random value within the specified tolerance range.
        
        This method generates a random value within the specified 
        tolerance range, clipping the value if the clip parameter is 
        set to True.
        
        Returns
        -------
        float
            The generated random value.
        """
        return getattr(self, self.dist)()
    
    def generate(self, n: int) -> NDArray:
        """Generate an array of random values.
        
        This method generates an array of random values by calling the 
        appropriate distribution method (normal or uniform) for the 
        specified number of times.
        
        Parameters
        ----------
        n : int
            The number of random values to generate.
        
        Returns
        -------
        NDArray[np.float_]
            An array of random values.
        """
        return np.array([self() for _ in range(n)])

def round_to_nearest(
        x: FloatOrArray,
        nearest: int = 5,
        digit: int = 3
        ) -> FloatOrArray:
    """Round to the nearest multiple of `nearest`.
    
    This function rounds the input data to the nearest multiple of the 
    specified `nearest` value at the specified `digit`.

    Parameters
    ----------
    x : float | NDArray[np.float_] | Series
        The input data to be rounded.
    nearest : int, optional
        The multiple to round to, by default 5.
    digit : int, optional
        The number of decimal places to round to, by default 3.
    
    Returns
    -------
    float | NDArray[np.float_] | Series
        The rounded data as the same type as the input.
    """
    factor = 10**digit
    return np.round(x / nearest * factor) / factor * nearest

def float_to_bins(
        data: NDArray | Series,
        num_bins: int,
        clip: bool = True,
        kind: Literal['linear', 'quantile'] = 'linear'
        ) -> NDArray[np.int_]: 
    """Convert an array of floats to a binned array of integers.

    This function bins the input data into the specified number of bins.
    If the kind parameter is set to 'linear', the bin edges are 
    calculated based on the minimum and maximum values of the input 
    data. If the kind parameter is set to 'quantile', the bin edges are 
    calculated based on the quantiles of the input data.
    Finally each float is assigned to the nearest bin using the 
    digitize function. If the clip parameter is set to True, the binned 
    data is clipped to the range [0, num_bins-1].
    
    Parameters
    ----------
    data : NDArray[np.float_] | Series
        The array of floats to be binned.
    num_bins : int
        The number of bins to use.
    clip : bool, optional
        Whether to clip the binned data to the range [0, num_bins-1], 
        by default True.
    kind : Literal['linear', 'quantile'], optional
        The method used to calculate the bin edges, by default 'linear'.
    
    Returns
    -------
    NDArray[np.int_]
        The binned array of integers.
    """
    assert kind in ['linear', 'quantile'], (
        f'kind must be either "linear" or "quantile", got "{kind}"')
    if kind == 'linear':
        bin_edges = np.linspace(np.min(data), np.max(data), num_bins + 1)
    else:
        bin_edges = np.quantile(data, np.linspace(0, 1, num_bins + 1))

    binned_data = np.digitize(data, bin_edges) - 1
    if clip:
        binned_data = np.clip(binned_data, 0, num_bins - 1)
    return binned_data

def precise_to_bin_nominals(
        precise_values: NDArray[np.float_] | Series,
        n_bins: int,
        distance: float | None = None,
        kind: Literal['linear', 'quantile'] = 'linear'
        ) -> NDArray[np.float_]:
    """Calculate the bin nominals

    Suppose we simulate the precise values we need, but we can't provide
    them continuously, but only at certain intervals with a certain 
    tolerance. This function is used to calculate the nominal values for 
    each of these intervals (bins).
    - If kind is 'linear', the bin nominals are evenly distributed at 
      the specified distance and aligned to the mean of precise values.
    - If kind is 'quantile', the bin nominals are calculated as evenly 
      spaced quantiles of the precise values.
    
    Parameters
    ----------
    precise_values : float
        The precise values.
    n_bins : int
        The number of bins.
    distance : float | None, optional
        The distance between the bins e.g. the process tolerance.
        Only considered if kind is "linear" and must be specified in 
        this case. Default is None.
    kind : Literal['linear', 'quantile'] = 'linear'
        The method used to calculate the bin nominals.
    
    Returns
    -------
    NDArray[np.float_]
        The bin nominals.
    """
    assert kind in ['linear', 'quantile'], (
        f'kind must be either "linear" or "quantile", got "{kind}"')
    
    if kind == 'linear':
        assert isinstance(distance, (int, float)), (
            f'Specify a distance between the bins, got {distance}')
        base = (
            distance / 2 
            + np.array([i * distance for i in range(n_bins)]))
        shift = np.mean(precise_values) - np.mean(base)
        nominals = base + shift
    else:
        quantiles = [i/(n_bins + 1) for i in range(1, n_bins + 1)]
        nominals = np.array([np.quantile(precise_values, q) for q in quantiles])
    return nominals

__all__ = [
    'SpecLimits',
    'Parameter',
    'RandomProcessValue',
    'round_to_nearest',
    'float_to_bins',
    'precise_to_bin_nominals',]
