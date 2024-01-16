import numpy as np

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Tuple
from typing import Literal
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .._constants import COLOR
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


class BaseCategoryLabelHandler(ABC):

    __slots__ = ('_categories', '_labels', '_n')
    _categories: Tuple[str]
    _labels: Tuple[str]
    _n: int
    
    def __init__(self, labels: Tuple) -> None:
        self._n = len(labels)
        self._labels = ()
        self.labels = labels

    @property
    def categories(self) -> Tuple[str]:
        return self._categories

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
        try:
            idx = self.labels.index(label)
        except ValueError:
            raise KeyError(f"Can't get category for label '{label}', got {self.labels}")
        return self.categories[idx]

    def __str__(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def handles_labels(self) -> Tuple[Tuple[Patch | Line2D], Tuple[str]]: ...


class HueLabelHandler(BaseCategoryLabelHandler):

    _categories = CATEGORY.COLORS

    def __init__(self, labels: Tuple) -> None:
        super().__init__(labels)
    
    @property
    def colors(self):
        return self.categories[:self.n_used]

    def handles_labels(self) -> Tuple[Tuple[Patch | Line2D], Tuple[str]]:
        handles = tuple(Patch(color=c, lw=0) for c in self.colors)
        return handles, self.labels


class ShapeLabelHandler(BaseCategoryLabelHandler):

    _categories = CATEGORY.MARKERS

    def __init__(self, labels: Tuple) -> None:
        super().__init__(labels)
    
    @property
    def markers(self):
        """Get used markers"""
        return self.categories[:self.n_used]

    def handles_labels(self) -> Tuple[Tuple[Patch | Line2D], Tuple[str]]:
        xy = [[], []]
        handles = tuple(
            Line2D(*xy, marker=m, c=COLOR.HANDLES) for m in self.markers)
        return handles, self.labels


class SizeLabelHandler(BaseCategoryLabelHandler):

    __slots__: ('_min', '_max', 'kind')
    _categories = CATEGORY.HANDLE_SIZES
    _min: int | float
    _max: int | float
    kind: Literal['scatter', 'line']

    def __init__(
            self, min_value: int | float, max_value: int | float,
            kind: Literal['scatter', 'line'] = 'scatter') -> None:
        assert kind in ['scatter', 'line']
        assert max_value > min_value
        self.kind = kind
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
        xy = [[], []]
        handles = tuple(
            Line2D(*xy, c=COLOR.HANDLES, markersize=s) for s in self.categories)
        return handles, self.labels
    
    def __getitem__(self, item: int | float) -> float:
        return self.sizes([item])[0]
    
    def sizes(self, values: ArrayLike) -> np.ndarray:
        """Convert values into size values for markers"""
        sizes = self.factor * (np.array(values) - self._min) + self.offset
        if self.kind == 'scatter':
            sizes = np.square(sizes)
        return sizes


__all__ = [
    add_second_axis.__name__,
    HueLabelHandler.__name__,
    ShapeLabelHandler.__name__,
    SizeLabelHandler.__name__,
    ]