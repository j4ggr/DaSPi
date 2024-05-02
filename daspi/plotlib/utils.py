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

from ..typing import LegendHandles
from ..constants import KW
from ..constants import PLOTTER
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

    __slots__ = ('_categories', '_default', '_labels', '_n')
    _categories: Tuple
    _default: Any
    _labels: Tuple
    _n: int
    
    def __init__(self, labels: Tuple) -> None:
        self._n = len(labels)
        self._default = None
        self.labels = labels

    @property
    def categories(self) -> Tuple:
        return self._categories

    @property
    def default(self) -> Any:
        """Get default category item"""
        if self._default is None:
            self._default = self.categories[0]
        return self._default

    @property
    def labels(self) -> Tuple[str]:
        return self._labels
    @labels.setter
    def labels(self, labels: Tuple[str]) -> None:
        assert self.n_used <= self.n_allowed, (
            f'{self} can handle {self.n_allowed} categories, got {len(labels)}')
        assert self.n_used == len(set(labels)), (
            'Labels occur more than once, only unique labels are allowed')
        self._labels = labels

    @property
    def n_used(self) -> int:
        """Get amount of used categories"""
        return self._n

    @property
    def n_allowed(self) -> int:
        """Allowed amount of categories"""
        return len(self.categories)
    
    def __getitem__(self, label: Any) -> Any:
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
        return self.__class__.__name__
    
    def __bool__(self) -> bool:
        return bool(self._n)
    
    @abstractmethod
    def handles_labels(self) -> Tuple[LegendHandles, Tuple[str, ...]]: ...


class HueLabel(_CategoryLabel):

    _categories: Tuple[str, ...] = CATEGORY.COLORS

    def __init__(self, labels: Tuple[str]) -> None:
        super().__init__(labels)
    
    @property
    def colors(self) -> Tuple[str, ...]:
        return self.categories[:self.n_used]

    def handles_labels(self) -> Tuple[LegendHandles, Tuple[str, ...]]:
        handles = tuple(Patch(color=c, **KW.HUE_HANDLES) for c in self.colors)
        return handles, self.labels


class ShapeLabel(_CategoryLabel):

    _categories: Tuple[str, ...] = CATEGORY.MARKERS

    def __init__(self, labels: Tuple) -> None:
        super().__init__(labels)
    
    @property
    def markers(self) -> Tuple[str, ...]:
        """Get used markers"""
        return self.categories[:self.n_used]

    def handles_labels(self) -> Tuple[LegendHandles, Tuple[str, ...]]:
        handles = tuple(
            Line2D(marker=m, **KW.SHAPE_HANDLES) for m in self.markers)
        return handles, self.labels


class SizeLabel(_CategoryLabel):

    __slots__ = ('_min', '_max')
    _categories: Tuple[int, ...] = CATEGORY.HANDLE_SIZES
    _min: int | float
    _max: int | float

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
        """Offset for value to size transformation"""
        return CATEGORY.MARKERSIZE_LIMITS[0]
    
    @property
    def factor(self) -> float:
        """Factor for value to size transformation"""
        low, upp = CATEGORY.MARKERSIZE_LIMITS
        return (upp - low)/(self._max - self._min)
    
    def handles_labels(self) -> Tuple[LegendHandles, Tuple[str, ...]]:
        handles = tuple(
            Line2D(markersize=s, **KW.SIZE_HANDLES) for s in self.categories)
        return handles, self.labels
    
    def __getitem__(self, item: int | float | None) -> float:
        if item is None:
            return self.default
        return self([item])[0]
    
    def __call__(self, values: ArrayLike) -> NDArray:
        """Convert values into size values for markers"""
        sizes = self.factor * (np.array(values) - self._min) + self.offset
        sizes = np.square(sizes)
        return sizes

    
class Dodger:

    __slots__ = (
        'categories', 'ticks', 'tick_lables', 'amount', 'width', 'dodge',
        '_default')
    categories: Tuple[str]
    ticks: NDArray[np.int_]
    tick_lables: Tuple[str]
    width: float
    amount: int
    dodge: Dict[str, float]
    _default: int

    def __init__(
            self, categories: Tuple[str], tick_labels: Tuple[str]) -> None:
        self.categories = categories
        self.tick_lables = tick_labels
        self.ticks = np.arange(len(tick_labels)) + PLOTTER.DEFAULT_F_BASE
        self.amount = max(len(self.categories), 1)
        space = CATEGORY.FEATURE_SPACE/self.amount
        self.width = space - CATEGORY.FEATURE_PAD

        offset = (space - CATEGORY.FEATURE_SPACE) / 2
        _dodge = tuple(i*space + offset for i in range(self.amount))
        self.dodge = {c: d for c, d in zip(self.categories, _dodge)}
        self._default = 0
    
    @property
    def lim(self) -> Tuple[float, float]:
        """Get the required axis limits."""
        return (np.min(self.ticks) - 0.5, np.max(self.ticks) + 0.5)
    
    def __getitem__(self, category: str | None) -> float | int:
        """Get the dodge value for given category"""
        if category is None:
            return self._default
        return self.dodge.get(category, self._default)
    
    def __call__(self, values: Series, category: str) -> pd.Series:
        """Replace source values to dodged ticks using given category"""
        if not self:
            return values
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        ticks = self.ticks + self[category]
        return values.replace(dict(zip(self.tick_lables, ticks)))
    
    def __bool__(self) -> bool:
        return len(self.categories) > 1

__all__ = [
    'shared_axes',
    'HueLabel',
    'ShapeLabel',
    'SizeLabel',
    'Dodger',
    ]
