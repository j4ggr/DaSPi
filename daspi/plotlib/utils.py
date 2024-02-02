import numpy as np
import pandas as pd

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .._constants import KW
from .._constants import CATEGORY


def add_second_axis(# TODO remove if not needed
        ax: Axes, which: Literal['x', 'y'], tick_labels: List[str], 
        axis_label: str = '') -> Axes:
    """Adds e second axis to the given direction, sharing the other one.
    
    Parameters
    ----------
    ax : Axes
        Base axes object to add second axis
    which : {'x', 'y'}
        From which axis a second one should be added
    tick_labels : list of str, optional
        Axis tick labels
    axis_label : str, optional
        Label for axis, corresponds to ylabel if which is 'y', else to
        xlabel, by default ''
    
    Returns
    -------
    ax2 : Axes
        The added axis object
    """
    ticks = [i for i in range(len(tick_labels))]
    ax2 = ax.twinx() if which == 'y' else ax.twiny()
    keys = (f'{which}{l}' for l in ('label', 'ticks', 'ticklabels'))
    values = (axis_label, ticks, tick_labels)
    ax2.set(**{k: v for k, v in zip(keys, values)})
    
    margins = ax.margins()
    ax2.margins(x=margins[0]/2, y=margins[1]/2)
    return ax2


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
    def labels(self) -> Tuple:
        return self._labels
    @labels.setter
    def labels(self, labels: Tuple):
        assert self.n_used <= self.n_allowed, f'{self} can handle {self.n_allowed} categories, got {len(labels)}'
        assert self.n_used == len(set(labels)), f'One or more labels occur more than once, only unique labels are allowed'
        self._labels = labels

    @property
    def n_used(self):
        """Get amount of used categories"""
        return self._n

    @property
    def n_allowed(self) -> int:
        """Allowed amount of categories"""
        return len(self.categories)
    
    def __getitem__(self, label: Any) -> str:
        if label is not None: 
            try:
                idx = self.labels.index(label)
            except ValueError:
                raise KeyError(f"Can't get category for label '{label}', got {self.labels}")
            item = self.categories[idx]
        else:
            item = self.default
        return item

    def __str__(self) -> str:
        return self.__class__.__name__
    
    def __bool__(self) -> str:
        return bool(self._n)
    
    @abstractmethod
    def handles_labels(self) -> Tuple[Tuple[Patch | Line2D], Tuple[str]]: ...


class HueLabel(_CategoryLabel):

    _categories: Tuple[str] = CATEGORY.COLORS

    def __init__(self, labels: Tuple[str]) -> None:
        super().__init__(labels)
    
    @property
    def colors(self):
        return self.categories[:self.n_used]

    def handles_labels(self) -> Tuple[Tuple[Patch | Line2D], Tuple[str]]:
        handles = tuple(Patch(color=c, **KW.HUE_HANDLES) for c in self.colors)
        return handles, self.labels


class ShapeLabel(_CategoryLabel):

    _categories: Tuple[str] = CATEGORY.MARKERS

    def __init__(self, labels: Tuple) -> None:
        super().__init__(labels)
    
    @property
    def markers(self):
        """Get used markers"""
        return self.categories[:self.n_used]

    def handles_labels(self) -> Tuple[Tuple[Patch | Line2D], Tuple[str]]:
        handles = tuple(
            Line2D(marker=m, **KW.SHAPE_HANDLES) for m in self.markers)
        return handles, self.labels


class SizeLabel(_CategoryLabel):

    __slots__: ('_min', '_max')
    _categories: Tuple[int] = CATEGORY.HANDLE_SIZES
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
    
    def handles_labels(self) -> Tuple[Tuple[Patch | Line2D], Tuple[str]]:
        handles = tuple(
            Line2D(markersize=s, **KW.SIZE_HANDLES) for s in self.categories)
        return handles, self.labels
    
    def __getitem__(self, item: int | float | None) -> float:
        if item is None: return self.default
        return self([item])[0]
    
    def __call__(self, values: ArrayLike) -> np.ndarray:
        """Convert values into size values for markers"""
        sizes = self.factor * (np.array(values) - self._min) + self.offset
        sizes = np.square(sizes)
        return sizes

    
class Dodger:

    __slots__ = (
        'categories', 'ticks', 'tick_lables', 'amount', 'width', 'dodge',
        '_default')
    categories: Tuple[str]
    ticks: Tuple[int]
    tick_lables: Tuple[str]
    width: float
    amount: int
    dodge: Dict[str, float]
    _default: int

    def __init__(
            self, categories: Tuple[str], tick_labels: Tuple[str]) -> None:
        self.categories = categories
        self.tick_lables = tick_labels
        self.ticks = np.arange(1, len(tick_labels) + 1)
        self.amount = max(len(self.categories), 1)
        space = CATEGORY.FEATURE_SPACE/self.amount
        self.width = space - CATEGORY.FEATURE_PAD

        offset = (space - CATEGORY.FEATURE_SPACE) / 2
        _dodge = tuple(i*space + offset for i in range(self.amount))
        self.dodge = {l: d for l, d in zip(self.categories, _dodge)}
        self._default = 0
    
    def __getitem__(self, category: str | None) -> float | int:
        """Get the dodge value for given category"""
        if category is None: return self._default
        return self.dodge.get(category, self._default)
    
    #TODO: does not work yet
    def __call__(self, values: pd.Series, category: str) -> pd.Series:
        """Replace source values to dodged ticks using given category"""
        if not self: return values
        ticks = self.ticks + self[category]
        return values.replace(dict(zip(self.tick_lables, ticks)))
    
    def __bool__(self) -> bool:
        return len(self.categories) > 1


__all__ = [
    add_second_axis.__name__,
    HueLabel.__name__,
    ShapeLabel.__name__,
    SizeLabel.__name__,
    Dodger.__name__,
    ]