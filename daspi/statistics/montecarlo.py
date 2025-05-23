import random

import numpy as np

from math import pi
from math import sin
from typing import Any
from typing import Self
from typing import Tuple
from typing import Literal
from dataclasses import dataclass
from numpy.typing import NDArray
from pandas.core.series import Series

from .._typing import FloatOrArray


@dataclass(frozen=True)
class SpecLimits:
    """Class to hold the limits of a parameter specification.

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
    def tolerance(self) -> float:
        """Tolerance range, returns the difference between upper and 
        lower limits. (read-only)"""
        return self.upper - self.lower
    
    @property
    def nominal(self) -> float:
        """Nominal value, returns the average of upper and lower limits
        (read-only)."""
        return (self.upper + self.lower) / 2
    
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
    

class Specification:
    """Class for defining the boundaries of a parameter specification. 
    
    This class is used to create constants. For this reason, all 
    properties are uppercase.

    Parameters
    ----------
    limits : SpecLimits | Tuple[float, float] | None, optional
        The limits of the specification. The limits are automatically 
        calculated if only tolerance is given. Default is None.
    tolerance : float or None, optional
        The tolerance range of the specification. Default is None.
    nominal : float or None, optional
        The nominal value of the specification. Default is None.
    n_digits : int, optional
        The number of decimal places to counteract the rounding error 
        of floating-point numbers. Default is 10.

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
    Create a specification with limits:
    ``` python
    Specification(limits=(0, 1))
    ```

    ```
    Specification(limits=(0, 1), tolerance=1.0, nominal=0.5)
    ```
    
    Create a specification with tolerance and nominal:
    ``` python
    Specification(tolerance=0.1, nominal=0.5)
    ```
    
    ```
    Specification(limits=(0.45, 0.55), tolerance=0.1, nominal=0.5)
    ```
    """

    __slots__ = ('_limits', '_tolerance', '_nominal', '_resolution')

    _limits: SpecLimits
    """The limits of the specification parameter."""

    _tolerance: float
    """The tolerance range of the specification."""

    _nominal: float
    """The nominal value of the specification."""

    _resolution: int
    """The resolution as 1e-n_digits."""

    def __init__(
            self,
            *,
            limits: SpecLimits | Tuple[float, float] | None = None, 
            tolerance: float | None = None,
            nominal: float | None = None,
            n_digits: int = 10
            ) -> None:
        assert n_digits > 0, 'n_digits must be greater than 0.'

        if isinstance(limits, tuple):
            limits = SpecLimits(*limits)
        elif limits is None:
            assert tolerance is not None and nominal is not None, (
                'Either limits or tolerance and nominal must be provided.')
            limits = SpecLimits(
                round(nominal - tolerance / 2, n_digits),
                round(nominal + tolerance / 2, n_digits))

        if tolerance is None:
            assert limits.are_both_finite, (
                'If tolerance is not provided, limits must be finite.')
            tolerance = round(limits.upper - limits.lower, n_digits)
        
        if nominal is None:
            assert limits.are_both_finite, (
                'If nominal is not provided, limits must be finite.')
            nominal = round(limits.lower + tolerance / 2, n_digits)

        self._limits = limits
        self._tolerance = tolerance
        self._nominal = nominal
        self._resolution = 1 ** -n_digits
    
    @property
    def LIMITS(self) -> SpecLimits:
        """The lower and upper limits of the specification (read-only)."""
        return self._limits
    
    @property
    def TOLERANCE(self) -> float:
        """The tolerance range of the specification."""
        return self._tolerance
    
    @property
    def NOMINAL(self) -> float:
        """The nominal value of the specification."""
        return self._nominal
    
    @property
    def RESOLUTION(self) -> float:
        """The resolution of the specification."""
        return self._resolution
    
    @property
    def LOWER(self) -> float:
        """The lower limit of the specification."""
        return self.LIMITS.lower
    
    @property
    def UPPER(self) -> float:
        """The upper limit of the specification."""
        return self.LIMITS.upper
    
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

    This class generates random values based on the specified
    distribution and parameters. The generated values are within the 
    specified limits and tolerance comi. The class supports three types 
    of distributions: normal, uniform, and circular. The generated
    values can be clipped to the range defined by the tolerance. The 
    class also provides a method to generate an array of random values.
    
    Parameters
    ----------
    specification : Specification
        The specification for which the random value will be generated.
    dist : {'normal', 'uniform', 'circular'}
        The distribution from which the random value will be generated.
    clip : bool, optional
        Whether to clip the generated value to the range defined by the
        tolerance. Default is False.
    
    Examples
    --------
    Generate a random value from a uniform distribution:
    
    ``` python
    import daspi as dsp

    PARAM_I = dsp.Specification(limits=(5, 10))
    PARAM_II = dsp.Specification(tolerance=0.1, nominal=3)

    rpv = RandomProcessValue(PARAM_I, 'uniform')
    value = rpv()
    ```

    Generate a array from a normal distribution with clipping:

    ```	python
    rpv = RandomProcessValue(PARAM_II, 'normal', clip=True)
    array = rpv.generate(100_000)
    ```
    """

    _allowed_dists: Tuple[str, ...] = (
        'normal', 'uniform', 'circular', 'coaxial', 'perpendicular')
    """Allowed distributions for the RandomProcessValue class."""

    def __init__(
            self,
            specification: Specification,
            dist: Literal['normal', 'uniform', 'circular', 'coaxial', 'perpendicular'],
            clip: bool = False,
            ) -> None:
        assert dist in self._allowed_dists, (
            f'Distribution must be one of {self._allowed_dists}, got {dist}.')
        self.specification = specification
        self.dist = dist
        self.clip_within_tolerance = clip
    
    @property
    def scale(self) -> float:
        """Scale factor for the distribution (6 sigma rule)."""
        return self.specification.TOLERANCE / 6
    
    @property
    def loc(self) -> float:
        """Location parameter of the distribution."""
        return self.specification.NOMINAL
    
    @property
    def lower(self) -> float:
        """Lower bound of the distribution."""
        return self.specification.LOWER
    
    @property
    def upper(self) -> float:
        """Upper bound of the distribution."""
        return self.specification.UPPER
    
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
        value = random.normalvariate(self.loc, self.scale)
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
        
        Returns
        -------
        float
            The modified value after applying the random circular offset.
        """
        return self.specification.TOLERANCE * sin(random.uniform(0, 2 * pi))
    
    def coaxial(self) -> float:
        """Generate a random coaxiality value.
        
        This method generates a random coaxiality value by applying a 
        circular distribution to the tolerance range. The amplitude of 
        the coaxiality is determined by a normal distribution within 
        the tolerance range, and the direction is randomized using a 
        uniform distribution between 0 and 2π radians.
        
        Returns
        -------
        float
            The generated random coaxiality value.
        """
        return self.normal() * sin(random.uniform(0, 2 * pi))
    
    def perpendicular(self) -> float:
        """Generate a random perpendicularity value.
        
        This method generates a random perpendicularity value by applying 
        a circular distribution to the tolerance range. The amplitude of 
        the perpendicularity is determined by a normal distribution 
        within the tolerance range, and the direction is randomized 
        using a uniform distribution between 0 and 2π radians. In 
        principle, this is exactly the same as coaxiality.
        
        Returns
        -------
        float
            The generated random perpendicularity value.
        
        Notes
        -----
        The returned value is in the same unit as the specified 
        tolerance (usually mm). When specifying tolerances according to 
        GPS, a protractor is held against the measuring object, and the 
        maximum gap must not be larger than the specified tolerance.
        For further information, see:
        https://www.keyence.de/ss/products/measure-sys/gd-and-t/orientation-tolerance/perpendicularity.jsp
        """
        return self.normal() * sin(random.uniform(0, 2 * pi))
    
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
        NDArray[np.float64]
            An array of random values.
        """
        return np.array([self() for _ in range(n)])


class Binning:
    """A class for binning precise values into a specified number of 
    bins.

    This class is designed to handle scenarios where precise values are 
    available only at certain intervals with a defined tolerance. It 
    provides methods to bin these values using either a linear or 
    quantile approach.

    The binning method is determined by the `kind` parameter:
    - If `kind` is set to 'linear', bin edges are calculated based on 
      the minimum and maximum values of the provided data.
    - If `kind` is set to 'quantile', bin edges are determined by the 
      quantiles of the input data.

    The `values` method assigns each data point to the nearest bin using 
    a digitizing function. For more accurate results, the 
    `round_to_nearest` method can be used to round the bin nominal 
    values.

    Parameters
    ----------
    data : NDArray[np.float64] | Series
        The precise values to be binned and used for calculations.
    num_bins : int
        The total number of bins to create.
    kind : Literal['linear', 'quantile'], optional
        The method to calculate the bin edges, default is 'quantile'.
    distance : float | None, optional
        The distance between the bins, representing the process tolerance.
        This parameter is only relevant when `kind` is 'linear' and must 
        be specified in that case. Default is None.

    Raises
    ------
    AssertionError
        If the `distance` is not specified when `kind` is 'linear'.
    
    Examples
    --------
    Example 1: Binning with linear approach
    ``` python
    import numpy as np
    import daspi as dsp
    data = np.array([1.5, 2.3, 3.7, 4.1, 5.0])
    binning = dsp.Binning(data, num_bins=3, distance=1.0, kind='linear')
    binned_values = binning.values()
    print(f'{binning.nominals=}')
    print(f'{binning.indices=}')
    print(binned_values)
    ```

    ``` console
    binning.nominals=array([2.32, 3.32, 4.32])
    binning.indices=array([0, 0, 1, 2, 2])
    [2.32, 2.32, 3.32, 4.32, 4.32]
    ```

    Example 2: Binning with quantile approach
    ``` python
    data = np.array([1.5, 2.3, 3.7, 4.1, 5.0])
    binning = dsp.Binning(data, num_bins=3, kind='quantile')
    binned_values = binning.values()
    print(f'{binning.nominals=}')
    print(f'{binning.indices=}')
    print(binned_values)
    ```

    ``` console
    binning.nominals=array([2.3, 3.7, 4.1])
    binning.indices=array([0, 0, 1, 2, 2])
    [2.3, 2.3, 3.7, 4.1, 4.1]
    ```

    Example 3: Rounding nominals
    ``` python
    data = np.array([1.5, 2.3, 3.7, 4.1, 5.0])
    binning = dsp.Binning(data, num_bins=3, kind='quantile')
    binning.round_to_nearest(nearest=5, digit=1)
    binned_values = binning.values()
    print(f'{binning.nominals=}')
    print(binned_values)
    ```

    ``` console
    binning.nominals=array([2.5, 3.5, 4.0])
    [2.5, 2.5, 3.5, 4.0, 4.0]
    ```
    """
    def __init__(
            self,
            data: NDArray[np.float64] | Series,
            num_bins: int,
            kind: Literal['linear', 'quantile'] = 'quantile',
            distance: float | None = None,
            ) -> None:
        self.data = np.asarray(data)
        self.num_bins = num_bins
        self.distance = distance
        self.kind = kind
        self.edges = self._calculate_edges()
        self.indices = self._calculate_indices()
        self.nominals = self._calculate_nominals()

    def _calculate_edges(self) -> np.ndarray:
        """Calculate the bin edges based on the specified method.

        Returns
        -------
        NDArray[np.float64]
            The calculated bin edges.
        """
        if self.kind == 'linear':
            edges = np.linspace(
                np.min(self.data),
                np.max(self.data),
                self.num_bins + 1)
        else:  # kind == 'quantile'
            edges = np.quantile(
                self.data,
                np.linspace(0, 1, self.num_bins + 1))
        return edges

    def _calculate_indices(self) -> np.ndarray:
        """Convert the precise values to bin indices.

        Returns
        -------
        NDArray[np.int_]
            The binned array of integers.
        """
        binned_data = np.digitize(self.data, self.edges) - 1
        return np.clip(binned_data, 0, self.num_bins - 1)

    def _calculate_nominals(self) -> NDArray[np.float64]:
        """Calculate the bin nominals based on the specified method.

        Returns
        -------
        NDArray[np.float64]
            The bin nominals.
        """
        if self.kind == 'linear':
            assert isinstance(self.distance, (int, float)), (
                f'Specify a distance between the bins, got {self.distance}')
            base = (
                self.distance / 2
                + np.array([i * self.distance for i in range(self.num_bins)]))
            shift = np.mean(self.data) - np.mean(base)
            nominals = base + shift
        else:
            _n = self.num_bins + 1
            quantiles = [i / _n for i in range(1, _n)]
            nominals = np.array(
                [np.quantile(self.data, q) for q in quantiles])
        return nominals

    def round_to_nearest(self, nearest: int = 5, digit: int = 3) -> Self:
        """Round to the nearest multiple of `nearest`.
        
        This function rounds the input data to the nearest multiple of 
        the specified `nearest` value at the specified `digit`.

        Parameters
        ----------
        nearest : int, optional
            The multiple to round to, by default 5.
        digit : int, optional
            The number of decimal places to round to, by default 3.
        
        Returns
        -------
        Self
            The instance with the rounded nominals.
        """
        self.nominals = round_to_nearest(self.nominals, nearest, digit)
        return self

    def values(self) -> NDArray[np.float64]:
        """Get the binned values based on the calculated indices and 
        nominals.

        Returns
        -------
        NDArray[np.float64]
            The binned values.
        """
        assert hasattr(self, 'nominals'), (
            'Calculate the nominals first using the calculate_nominals method')
        return self.nominals[self.indices]


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
    x : float | NDArray[np.float64] | Series
        The input data to be rounded.
    nearest : int, optional
        The multiple to round to, by default 5.
    digit : int, optional
        The number of decimal places to round to, by default 3.
    
    Returns
    -------
    float | NDArray[np.float64] | Series
        The rounded data as the same type as the input.
    """
    factor = 10**digit
    return np.round(x / nearest * factor) / factor * nearest

def inclination_displacement(
        perpendicularity: FloatOrArray,
        height: float,
        distance: float
        ) -> FloatOrArray:
    """Calculate the displacement from a distant point due to 
    perpendicularity deviation. Perpendicularity is measured in mm. To 
    calculate the angle, the height at which perpendicularity is 
    measured is required as the maximum distance in the drawing.

    $$ displacement = distance * sin(arctan(perpendicularity/height)) $$
    
    Parameters
    ----------
    perpendicularity : float | NDArray[np.float64] | Series
        The perpendicularity of the surface.
    distance : float
        The distance from the point to the surface.

    Returns
    -------
    float | NDArray[np.float64] | Series
        The displacement of the point.
    """
    displacement = distance * np.tan(np.arctan(perpendicularity/height))
    return displacement # type: ignore

__all__ = [
    'SpecLimits',
    'Specification',
    'RandomProcessValue',
    'Binning',
    'round_to_nearest',
    'inclination_displacement']
