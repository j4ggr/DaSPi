import numpy as np
import pandas as pd

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import Iterable
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from pandas.core.series import Series
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats._distn_infrastructure import rv_continuous

from .._constants import KW
from .._constants import DIST
from .._constants import CATEGORY
from ..statistics.estimation import Estimator


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
        assert self.n_used <= self.n_allowed, (
            f'{self} can handle {self.n_allowed} categories, got {len(labels)}')
        assert self.n_used == len(set(labels)), (
            f'Labels occur more than once, only unique labels are allowed')
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
                raise KeyError(
                    f"Can't get category for label '{label}', got {self.labels}")
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

    __slots__ = ('_min', '_max')
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
    
    def __call__(self, values: Series, category: str) -> pd.Series:
        """Replace source values to dodged ticks using given category"""
        if not self: return values
        if not isinstance(values, pd.Series): values = pd.Series(values)
        ticks = self.ticks + self[category]
        return values.replace(dict(zip(self.tick_lables, ticks)))
    
    def __bool__(self) -> bool:
        return len(self.categories) > 1


class Line:

    __slots__ = (
        'estimation', 'mean', 'median', 'control_limits', 'spec_limits')
    estimation: Estimator
    mean: bool
    median: bool
    control_limits: bool
    spec_limits: Tuple[float | int]
    
    def __init__(
        self,
        target: Iterable,
        mean: bool = False,
        median: bool = False,
        control_limits: bool = False,
        spec_limits: Tuple[float | int] = (), 
        strategy: Literal['fit', 'eval', 'norm', 'data'] = 'norm', 
        possible_dists: Tuple[str | rv_continuous] = DIST.COMMON,
        tolerance: float | int = 6
        ) -> None:
        self.estimation = Estimator(data=target, strategy=strategy)
    

    # __slots__ = (
    #     'strategy', 'tolerance', 'possible_dists', 'sample_color', '_mask',
    #     '_n', '_k', '_q_low', '_q_upp')
    # strategy: str
    # tolerance: int
    # possible_dists: Tuple[str | rv_continuous]
    # sample_color: bool
    # _mask: Tuple[bool]
    # _n: int
    # _k: int | None
    # _q_low: float | None
    # _q_upp: float | None

    # def __init__(
    #         self,
    #         source: DataFrame,
    #         target: str,
    #         strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
    #         tolerance: float | int = 6,
    #         possible_dists: Tuple[str | rv_continuous] = DIST.COMMON,
    #         show_mean: bool = True,
    #         show_median: bool = True,
    #         show_climits: bool = True,
    #         sample_color: bool = False,
    #         target_on_y: bool = True,
    #         color: str | None = None,
    #         ax: Axes | None = None,
    #         **kwds) -> None:
    #     self.strategy = strategy
    #     self.tolerance = tolerance
    #     self.possible_dists = possible_dists
    #     self.sample_color = sample_color
    #     self._mask = (show_mean, show_median, show_climits, show_climits)
    #     self._n = 100
    #     self._k = None
    #     self._q_low = None
    #     self._q_upp = None
    #     super().__init__(
    #         source=source, target=target, target_on_y=target_on_y, color=color,
    #         ax=ax)
    
    # @property
    # def _d(self) -> int:
    #     """Get decimals to format labels"""
    #     if self._n > 50: return 1
    #     if self._n > 5: return 2
    #     if self._n > 0.5: return 3
    #     return 4

    # @property
    # def kwds(self) -> Tuple[dict]:
    #     """Get keyword arguments for all lines that are plotted"""
    #     kwds = self._filter(
    #         (KW.MEAN_LINE, KW.MEDIAN_LINE, KW.CONTROL_LINE, KW.CONTROL_LINE))
    #     if self.sample_color:
    #         kwds = (kw | {'color': self.color} for kw in kwds)
    #     return kwds
    
    # @property
    # def names(self) -> Tuple[str]:
    #     """Get the column names for all lines that are plotted"""
    #     names = self._filter(
    #         (PLOTTER.MEAN, PLOTTER.MEDIAN, PLOTTER.LCL, PLOTTER.UCL))
    #     return names
    
    # @property
    # def attrs(self) -> Tuple[str]:
    #     attrs = self._filter(('mean', 'median', 'lcl', 'ucl'))
    #     return attrs
    
    # @property
    # def labels(self) -> Tuple[str]:
    #     """Get labels for lines"""
    #     if self.strategy == 'norm':
    #         lcl = r'\bar x-' + f'{self._k}' + r'\sigma'
    #         ucl = r'\bar x+' + f'{self._k}' + r'\sigma'
    #     else:
    #         lcl = r'x_{' + f'{self._q_low:.4f}' + '}'
    #         ucl = r'x_{' + f'{self._q_upp:.4f}' + '}'
    #     labels = self._filter((r'\bar x', r'x_{0.5}', lcl, ucl))
    #     if len(self.source) == 1:
    #         values = self._filter(tuple(self.source.loc[0, list(self.names)]))
    #         labels = (f'{l}={v:.{self._d}}' for l, v in zip(labels, values))
    #     labels = tuple(f'${l}$' for l in labels)
    #     return labels
    
    # def _filter(self, values: tuple) -> tuple:
    #     """Filter given values according to self._mask"""
    #     return tuple(v for v, m in zip(values, self._mask) if m)

    # def transform(
    #         self, feature_data: float | int, target_data: Series) -> DataFrame:
    #     estimation = Estimator(
    #         data=target_data, strategy=self.strategy, tolerance=self.tolerance,
    #         possible_dists=self.possible_dists)
    #     if estimation.n_samples < self._n: self._n = estimation.n_samples
    #     if self._k is None: self._k = int(estimation._k)
    #     if self._q_low is None: self._q_low = estimation.q_low
    #     if self._q_upp is None: self._q_low = estimation.q_upp
        
    #     data = pd.DataFrame(
    #         {self.feature: [feature_data]} | {
    #         n: [getattr(estimation, a)] for n, a in zip(self.names, self.attrs)})
    #     return data
    
    # def __call__(self):
    #     for kwds, name in zip(self.kwds, self.names):
    #         value = self.source[name][0]
    #         if self.target_on_y:
    #              self.ax.axhline(value, **kwds)
    #         else:
    #             self.ax.axvline(value, **kwds)

__all__ = [
    add_second_axis.__name__,
    HueLabel.__name__,
    ShapeLabel.__name__,
    SizeLabel.__name__,
    Dodger.__name__,
    ]