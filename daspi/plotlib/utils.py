import numpy as np
import pandas as pd

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from pandas.core.series import Series
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .._typing import LegendHandlesLabels
from ..constants import KW
from ..constants import DEFAULT
from ..constants import CATEGORY


def shared_axes(
        ax: Axes, which: Literal['x', 'y'], exclude: bool = True
        ) -> List[bool]:
    """Get all the axes from the figure of the given `ax` and compare 
    whether the `ax` share the given axis. 
    Get a map of boolean values as a list where all are `True` when the 
    axis is shared.
    
    Parameters
    ----------
    ax : Axes
        Base axes object to add second axis
    which : {'x', 'y'}
        From which axis a second one should be added
    exclude : bool, optional
        If True excludes the given `ax` in the returned map
    
    Returns
    -------
    List[bool]
        Flat map for axes that shares same axis
    """
    assert which in ('x', 'y')
    view = getattr(ax, f'get_shared_{which}_axes')()
    axes = [_ax for _ax in ax.figure.axes] if exclude else ax.figure.axes  # type: ignore
    return [view.joined(ax, _ax) for _ax in axes]


class _CategoryLabel(ABC):
    """Abstract base class representing a category label handler for
    plotted categorical values.

    Parameters
    ----------
    labels : Tuple[str, ...]
        Labels corresponding to the categories.
    """

    __slots__ = ('_categories', '_default', '_labels', '_n')

    _categories: Tuple
    """Tuple containing the available categories."""
    _default: Any
    """Default category item."""
    _labels: Tuple[str, ...]
    """Labels corresponding to the categories."""
    _n: int
    """Number of used categories."""
    
    def __init__(self, labels: Tuple[str, ...]) -> None:
        self._n = len(labels)
        self._default = None
        self.labels = labels

    @property
    def categories(self) -> Tuple:
        """Get the available categories (read-only)."""
        return self._categories

    @property
    def default(self) -> Any:
        """Get default category item (read-only)."""
        if self._default is None:
            self._default = self.categories[0]
        return self._default

    @property
    def labels(self) -> Tuple[str, ...]:
        """Get and set the labels corresponding to the categories."""
        return self._labels
    @labels.setter
    def labels(self, labels: Tuple[str, ...]) -> None:
        assert self.n_used <= self.n_allowed, (
            f'{self} can handle {self.n_allowed} categories, got {len(labels)}')
        assert self.n_used == len(set(labels)), (
            'Labels occur more than once, only unique labels are allowed')
        self._labels = labels

    @property
    def n_used(self) -> int:
        """Get the number of used categories (read-only)."""
        return self._n

    @property
    def n_allowed(self) -> int:
        """Get the allowed amount of categories (read-only)."""
        return len(self.categories)
    
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
            try:
                idx = self.labels.index(label)
            except ValueError:
                raise KeyError(
                    f"Can't get category for label '{label}', got {self.labels}")
            item = self.categories[idx]
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
    labels : Tuple[str]
        Labels corresponding to the hue categories.
    """

    _categories: Tuple[str, ...] = DEFAULT.PALETTE
    """Tuple containing the available hue categories as the current
    color palette"""


    def __init__(self, labels: Tuple[str]) -> None:
        super().__init__(labels)
    
    @property
    def colors(self) -> Tuple[str, ...]:
        """Get the available hue colors (read-only)."""
        return self.categories[:self.n_used]

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
    labels : Tuple
        Labels corresponding to the shape marker categories.
    """

    _categories: Tuple[str, ...] = CATEGORY.MARKERS
    """Tuple containing the available shape markers."""

    def __init__(self, labels: Tuple) -> None:
        super().__init__(labels)
    
    @property
    def markers(self) -> Tuple[str, ...]:
        """Get the used shape markers (read-only)."""
        return self.categories[:self.n_used]

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
    """

    __slots__ = ('_min', '_max')

    _categories: Tuple[int, ...] = CATEGORY.HANDLE_SIZES
    """Tuple containing the available marker sizes."""
    _min: int | float
    """Minimum value for the size range."""
    _max: int | float
    """Maximum value for the size range."""

    def __init__(
            self, min_value: int | float, max_value: int | float,
            ) -> None:
        assert max_value > min_value
        self._min = min_value
        self._max = max_value
        _int = isinstance(self._min, int) and isinstance(self._max, int)
        labels = np.linspace(
                self._min, self._max, CATEGORY.N_SIZE_BINS, 
                dtype = int if _int else float)
        if _int:
            labels = tuple(map(str, labels))
        else:
            labels = tuple(map(lambda x: f'{x:.3f}', labels))
        super().__init__(labels)
    
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
        return self.dodge.get(category, self._default)
    
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
    'shared_axes',
    'HueLabel',
    'ShapeLabel',
    'SizeLabel',
    'Dodger',
    ]
