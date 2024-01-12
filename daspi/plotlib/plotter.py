import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from typing import Literal
from typing import Hashable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import rgb2hex

from .._strings import STR
from ..statistics.estimation import estimate_kernel_density


class BasePlotter(ABC):

    __slots__ = (
        'source', 'y', 'x', 'color', 'orientation', 'ax', 'title', 'y_label', 
        'x_label')
    source: Hashable
    y: str
    s: str
    color: str
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
        self.color = color
        
    @property
    def transposed(self) -> bool:
        """True if orientation is 'horizontal'"""
        return self.orientation == 'horizontal'
        
    def set_labels(self):
        """Set x label, y label and title for the axes"""
        keys = ('title', 'xlabel', 'ylabel')
        values = (self.title, self.x_label, self.y_label)
        self.ax.set(**{k: v for k, v in zip(keys, values) if v is not None})

    @abstractmethod
    def plot(self):
        self.color = self.ax.get_lines()[-1].get_color()
        self.set_labels()


class Scatter(BasePlotter):

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
        collection = self.ax.scatter(self.x, self.y, c=self.color)
        self.color = rgb2hex(collection.get_facecolor())
        self.set_labels()


class Line(BasePlotter):

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
        super().plot()


class KDE(Line):

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
