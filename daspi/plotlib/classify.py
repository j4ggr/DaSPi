"""The provided Python code defines several classes and a list of
strings, all related to handling categorical features in data
visualization. Here's a brief description of each component:

1. _CategoryLabel: An abstract base class representing a category label 
handler for plotted categorical values. It has properties for available 
categories, default category, labels, and number of used categories. It
also includes methods for getting category items, retrieving legend
handles and labels, and checking if the object is empty.
2. HueLabel: A class representing category labels for hue values in
plots. It inherits from _CategoryLabel and adds a property for available
hue colors. It also overrides the handles_labels method to provide
specific legend handles and labels for hue categories.
3. ShapeLabel: A class representing category labels for shape markers in
plots. It inherits from _CategoryLabel and overrides the handles_labels
method to provide specific legend handles and labels for shape markers.
4. SizeLabel: A class representing category labels for marker sizes in
plots. It inherits from _CategoryLabel and adds properties for the
offset and factor for value-to-size transformation. It also overrides
the handles_labels method to provide specific legend handles and labels
for marker sizes. Additionally, it includes methods for getting size
values and converting values into size values for markers.
5. Dodger: A class for handling dodging of categorical features in
plots. It has properties for categories, ticks, tick labels, width of
each category bar, number of categories, dodge values, and default dodge
value. It also includes methods for getting dodge values, replacing
source values with dodged ticks, and checking if the object has more
than one category.
"""

import numpy as np
import pandas as pd

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Sequence
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from pandas._typing import Scalar
from pandas.core.series import Series
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .._typing import LegendHandlesLabels
from ..constants import KW
from ..constants import DEFAULT
from ..constants import CATEGORY


class _CategoryLabel(ABC):
    """Abstract base class representing a category label handler for
    plotted categorical values.

    Parameters
    ----------
    labels : Sequence[Scalar]
        Labels corresponding to the categories.
    """

    __slots__ = ('_default', '_categories', '_labels', '_n')

    _default: Any
    """Default category item."""
    _categories: Tuple[Any, ...]
    """Available categories."""
    _labels: Tuple[str, ...]
    """Labels corresponding to the categories."""
    _n: int
    """Number of used categories."""
    
    def __init__(
            self,
            labels: Sequence[Scalar],
            categories: Tuple[Any, ...]) -> None:
        self._n = len(labels)
        self._default = None
        self._categories = categories
        self.labels = labels

    @property
    def categories(self) -> Tuple[Any, ...]:
        """Get the available categories (read-only)."""
        return self._categories[:self.n_used]

    @property
    def default(self) -> Any:
        """Get default category item (read-only)."""
        if self._default is None:
            self._default = self._categories[0]
        return self._default

    @property
    def labels(self) -> Tuple[str, ...]:
        """Get and set the labels corresponding to the categories."""
        return self._labels
    @labels.setter
    def labels(self, labels: Sequence[Scalar]) -> None:
        assert self.n_used <= self.n_allowed, (
            f'{self} can handle {self.n_allowed} categories, got {len(labels)}')
        assert self.n_used == len(set(labels)), (
            'Labels occur more than once, only unique labels are allowed')
        self._labels = tuple(map(str, labels))

    @property
    def n_used(self) -> int:
        """Get the number of used categories (read-only)."""
        return self._n

    @property
    def n_allowed(self) -> int:
        """Get the allowed amount of categories (read-only)."""
        return len(self._categories)
    
    def __getitem__(self, label: Any) -> Any:
        """Get the category item corresponding to the given label.

        Parameters
        ----------
        label : Any
            The label for which to retrieve the category item.

        Returns
        -------
        Any
            The category item corresponding to the label.
        """
        if label is not None:
            _label = str(label)
            assert _label in self.labels, (
                f"Can't get category for label {_label}, got {self.labels}")
            item = self.categories[self.labels.index(_label)]
        else:
            item = self.default
        return item

    def __str__(self) -> str:
        """Get a string representation of the _CategoryLabel object."""
        return self.__class__.__name__
    
    def __bool__(self) -> bool:
        """Check if the _CategoryLabel object is non-empty."""
        return bool(self._n)
    
    @abstractmethod
    def handles_labels(self) -> LegendHandlesLabels:
        """Abstract method to retrieve legend handles and labels.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        pass


class HueLabel(_CategoryLabel):
    """A class representing category labels for hue values in plots.

    Parameters
    ----------
    labels : Sequence[Scalar]
        Labels corresponding to the hue categories.
    colors : Tuple[str, ...], optional
        Tuple of available hue categories as hex or str colors,
        by default `CATEGORY.PALETTE`.
    """

    def __init__(
            self,
            labels: Sequence[Scalar],
            colors: Tuple[str, ...]) -> None:
        super().__init__(labels, colors)
    
    @property
    def categories(self) -> Tuple[str, ...]:
        """Tuple containing the available hue categories as the current
        color palette as defined in constants `CATEGORY.MARKERS` or
        the given colors during initialization (read-only)."""
        return super().categories

    @property
    def colors(self) -> Tuple[str, ...]:
        """Get the available hue colors (read-only)."""
        return self.categories

    def handles_labels(self) -> LegendHandlesLabels:
        """Retrieve legend handles and labels for hue categories.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        handles = tuple(Patch(color=c, **KW.HUE_HANDLES) for c in self.colors)
        return handles, self.labels


class ShapeLabel(_CategoryLabel):
    """A class representing category labels for shape markers in plots.

    Parameters
    ----------
    labels : Sequence[Scalar]
        Labels corresponding to the shape marker categories.
    markers : Tuple[str, ...], optional
        Tuple of available shape marker categories as strings,
        by default `CATEGORY.MARKERS`.
    """

    def __init__(
            self,
            labels: Sequence[Scalar],
            markers: Tuple[str, ...]) -> None:
        super().__init__(tuple(map(str, labels)), markers)
    
    @property
    def categories(self) -> Tuple[str, ...]:
        """Tuple containing the available shape markers as defined
        in constants `CATEGORY.MARKERS` or the given markers during
        initialization (read-only)."""
        return super().categories
    
    @property
    def markers(self) -> Tuple[str, ...]:
        """Get the used shape markers (read-only)."""
        return self.categories

    def handles_labels(self) -> LegendHandlesLabels:
        """Retrieve legend handles and labels for shape markers.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        handles = tuple(
            Line2D(marker=m, **KW.SHAPE_HANDLES) for m in self.markers)
        return handles, self.labels


class SizeLabel(_CategoryLabel):
    """A class representing category labels for marker sizes in plots.

    Parameters
    ----------
    min_value : int | float
        Minimum value for the size range.
    max_value : int | float
        Maximum value for the size range.
    n_bins : int, optional
        Number of bins for the size range, by default 
        `CATEGORY.N_SIZE_BINS`.
    """

    __slots__ = ('_min', '_max')

    _min: int | float
    """Minimum value for the size range."""
    _max: int | float
    """Maximum value for the size range."""

    def __init__(
            self, 
            min_value: int | float,
            max_value: int | float,
            n_bins: int) -> None:
        assert max_value > min_value
        self._min = min_value
        self._max = max_value
        use_integer = isinstance(self._min, int) and isinstance(self._max, int)
        labels = np.linspace(
                self._min, self._max, CATEGORY.N_SIZE_BINS, 
                dtype = int if use_integer else float)
        if use_integer:
            labels = tuple(map(str, labels))
        else:
            labels = tuple(map(lambda x: f'{x:.3f}', labels))
        handle_sizes = tuple(np.linspace(
            *CATEGORY.MARKERSIZE_LIMITS, n_bins, dtype=int))
        super().__init__(labels, handle_sizes)
    
    @property
    def categories(self) -> Tuple[int, ...]:
        """Tuple containing the available marker sizes (read-only)."""
        return super().categories
    
    @property
    def offset(self) -> int:
        """Get the offset for value-to-size transformation (read-only)."""
        return CATEGORY.MARKERSIZE_LIMITS[0]

    @property
    def factor(self) -> float:
        """Get the factor for value-to-size transformation (read-only)."""
        low, upp = CATEGORY.MARKERSIZE_LIMITS
        return (upp - low)/(self._max - self._min)
    
    def handles_labels(self) -> LegendHandlesLabels:
        """Retrieve legend handles and labels for marker sizes.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        handles = tuple(
            Line2D(markersize=s, **KW.SIZE_HANDLES) for s in self.categories)
        return handles, self.labels
    
    def __getitem__(self, item: int | float | None) -> float:
        """Get the size value corresponding to the given item.

        Parameters
        ----------
        item : int | float | None
            The item for which to retrieve the size value.

        Returns
        -------
        float
            The size value corresponding to the item."""
        if item is None:
            return self.default
        return self([item])[0]
    
    def __call__(self, values: ArrayLike) -> NDArray:
        """Convert values into size values for markers.

        Parameters
        ----------
        values : ArrayLike
            Values to be converted into marker sizes.

        Returns
        -------
        NDArray
            Size values for markers.
        """
        sizes = self.factor * (np.array(values) - self._min) + self.offset
        sizes = np.square(sizes)
        return sizes


class Dodger:
    """A class for handling dodging of categorical features in plots.

    Parameters
    ----------
    categories : Tuple[str, ...]
        Categories corresponding to the features.
    tick_labels : Tuple[str, ...]
        Labels for the ticks on the axis.
    """

    __slots__ = (
        'categories', 'ticks', 'tick_lables', 'amount', 'width', 'dodge',
        '_default')
    
    categories: Tuple[str, ...]
    """Categories corresponding to the features."""
    ticks: NDArray[np.int_]
    """Numeric positions of the ticks."""
    tick_lables: Tuple[str, ...]
    """Labels for the ticks on the axis."""
    width: float
    """Width of each category bar."""
    amount: int
    """Number of categories."""
    dodge: Dict[str, float]
    """Dictionary mapping categories to dodge values."""
    _default: int
    """Default dodge value."""

    def __init__(
            self,
            categories: Tuple[str, ...],
            tick_labels: Tuple[str, ...]) -> None:
        self.categories = categories
        self.tick_lables = tick_labels
        self.ticks = np.arange(len(tick_labels)) + DEFAULT.FEATURE_BASE
        self.amount = max(len(self.categories), 1)
        space = CATEGORY.FEATURE_SPACE/self.amount
        self.width = space - CATEGORY.FEATURE_PAD
        offset = (space - CATEGORY.FEATURE_SPACE) / 2
        _dodge = tuple(i*space + offset for i in range(self.amount))
        self.dodge = {c: d for c, d in zip(self.categories, _dodge)}
        self._default = 0
    
    @property
    def lim(self) -> Tuple[float, float]:
        """Get the required axis limits (read-only)."""
        return (min(self.ticks) - 0.5, max(self.ticks) + 0.5)
    
    def __getitem__(self, category: str | None) -> float | int:
        """Get the dodge value for a given category.

        Parameters
        ----------
        category : str | None
            The category for which to retrieve the dodge value.

        Returns
        -------
        float | int
            The dodge value corresponding to the category.
        """
        if category is None:
            return self._default
        return self.dodge.get(str(category), self._default)
    
    def __call__(self, values: Series, category: str) -> pd.Series:
        """Replace source values with dodged ticks using the given category.

        Parameters
        ----------
        values : Series
            Source values to be replaced.
        category : str
            The category for which to apply the dodge.

        Returns
        -------
        pd.Series
            Series with replaced values.
        """
        if not self:
            return values
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        ticks = self.ticks + self[category]
        return values.replace(dict(zip(self.tick_lables, ticks)))
    
    def __bool__(self) -> bool:
        """Check if the Dodger object has more than one category.

        Returns
        -------
        bool
            True if there are multiple categories, False otherwise.
        """
        return len(self.categories) > 1

__all__ = [
    'HueLabel',
    'ShapeLabel',
    'SizeLabel',
    'Dodger',
    ]
