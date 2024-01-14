import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import Hashable
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .facets import LabelFacets
from .facets import CategoricalAxesFacets

from .._strings import STR
from .._constants import COLOR
from ..statistics.estimation import estimate_kernel_density


class BasePlotter(ABC):

    __slots__ = (
        'source', 'y', 'x', '_color', 'orientation', 'ax', 'title', 'y_label', 
        'x_label')
    source: Hashable
    y: str
    s: str
    _color: str
    orientation: str
    fig: Figure
    ax: Axes
    title: str
    y_label: str
    x_label: str

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            orientation : Literal['vertical', 'horizontal'] = 'vertical',
            color: str | None = None,
            ax: Axes | None = None, 
            title: str | None= '',
            target_label: str | None = None, 
            feature_label: str | None = None
            ) -> None:
        assert orientation in ['vertical', 'horizontal']
        self.orientation = orientation
        
        self.source = source
        if not feature:
            feature = 'feature'
            self.source[feature] = range(len(source[target]))
        self.y = target
        self.x = feature

        self.y_label = target if target_label is None else target_label
        self.x_label = feature if feature_label is None else feature_label
        self.title = title
        
        if self.transposed:
            self.x, self.y = self.y, self.x
            self.x_label, self.y_label = self.y_label, self.x_label
        
        self.fig, self.ax = plt.subplots(1, 1) if ax is None else ax.figure, ax
        self._color = color
        
    @property
    def transposed(self) -> bool:
        """True if orientation is 'horizontal'"""
        return self.orientation == 'horizontal'
    
    @property
    def color(self) -> str | None:
        """Get color of drawn artist"""
        if self._color is None:
            self._color = COLOR.PALETTE[0]
        return self.color
        
    def set_labels(self):
        """Set x label, y label and title for the axes"""
        keys = ('title', 'xlabel', 'ylabel')
        values = (self.title, self.x_label, self.y_label)
        self.ax.set(**{k: v for k, v in zip(keys, values) if v is not None})

    @abstractmethod
    def plot(self):...


class BaseScatter(BasePlotter):

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '', 
            orientation: Literal['vertical', 'horizontal'] = 'vertical', 
            color: str | None = None,
            ax: Axes | None = None,
            title: str = '', 
            target_label: str | None = None,
            feature_label: str | None = None
            ) -> None:
        super().__init__(
            source, target, feature, orientation, color, ax, title, 
            target_label, feature_label)
    
    def plot(self):
        self.ax.scatter(self.x, self.y, c=self.color)
        self.set_labels()


class BaseLine(BasePlotter):

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '', 
            orientation: Literal['vertical', 'horizontal'] = 'vertical', 
            color: str | None = None,
            ax: Axes | None = None,
            title: str = '', 
            target_label: str | None = None,
            feature_label: str | None = None
            ) -> None:
        super().__init__(
            source, target, feature, orientation, color, ax, title, 
            target_label, feature_label)
    
    def plot(self, **kw_line):
        kwds = {'c': self.color} | kw_line
        self.ax.plot(self.x, self.y, **kwds)
        self.set_labels()


class BaseKDE(BaseLine):

    __slots__ = ('base')
    base: float

    def __init__(
            self,
            source: Hashable,
            target: str,
            shift: float = 0,
            height: float | None = None,
            orientation: Literal['vertical', 'horizontal'] = 'vertical',
            color: str | None = None,
            ax: Axes | None = None,
            title: str | None = '',
            target_label: str | None = None,
            feature_label: str | None = None) -> None:
        
        feature = 'kde'
        if feature_label is None: 
            feature_label = STR['kde_ax_label']
        
        sequence, estimation = estimate_kernel_density(
            source[target], height, shift)
        data = pd.DataFrame({target: sequence, feature: estimation})
        
        self.base = shift
        super().__init__(
            data, target, feature, orientation, color, ax, title, 
            target_label, feature_label)
        
    def plot(self, kw_line: dict, **fill_kw):
        super().plot(**kw_line)
        if self.transposed:
            self.ax.fill_betweenx(self.y, self.base, self.x, **fill_kw)
        else:
            self.ax.fill_between(self.x, self.base, self.y)


class MultiVariablePlotter:

    marker_size_min: int = 2
    marker_size_max: int = 10
    n_marker_sizes: 4

    def __init__(
            self,
            plotter: BasePlotter,
            source: pd.DataFrame,
            target: str,
            feature: str = '',
            hue: str = '',
            shape: str = '',
            size: str = '',
            col: str = '',
            row: str = '',
            orientation: Literal['vertical', 'horizontal'] = 'vertical',
            feature_as_cat: bool = False) -> None:
        self.plotter: BasePlotter = plotter
        self.source: pd.DataFrame = source
        self.target: str = target
        self.feature: str = feature
        self.shape: str = shape
        self.hue: str = hue
        self.size: str = size
        self.col: str = col
        self.row: str = row
        self.orientation: Literal['vertical', 'horizontal'] = orientation
        self.feature_as_cat: bool = feature_as_cat
        self.axes_facets: CategoricalAxesFacets = CategoricalAxesFacets(
            source=self.source, col=self.col, row=self.row)
        self.label_facets: LabelFacets = LabelFacets()
        self.hue_labels: Tuple | None = self._categorical_labels_(hue)
        self.shape_labels: Tuple | None = self._categorical_labels_(shape)
        self.colors: Dict = {h: c for h, c in zip(self.hue_labels, COLOR.PALETTE)}
        self._groupers: Tuple[str] = (self.row, self.col, self.hue, self.shape)
        self._hue_label: str | None = None
        self._shape_label: str | None = None

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements"""
        return self.axes_facets.figure
    
    @property
    def axes(self) -> ArrayLike[Axes]:
        """Get the created axes"""
        return self.axes_facets.axes
    
    @property
    def groupers(self) -> List[str]:
        return [g for g in self._groupers if g]

    def _categorical_labels_(self, colname: str) -> Tuple | None:
        """Get sorted unique elements of given column in source"""
        return self.axes_facets._categorical_labels_(colname)
    
    def plot(self):
        self.source.groupby(self.groupers)
