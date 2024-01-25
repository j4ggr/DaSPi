import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from typing import Literal
from typing import Hashable
from typing import Iterable
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._strings import STR
from .._constants import COLOR
from ..statistics.estimation import estimate_kernel_density


class BasePlotter(ABC):

    __slots__ = (
        'source', 'y', 'x', '_color', 'orientation', 'ax', 'kind')
    source: Hashable
    y: str
    s: str
    _color: str
    orientation: str
    fig: Figure
    ax: Axes
    kind: Literal['scatter', 'line']

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            orientation : Literal['vertical', 'horizontal'] = 'vertical',
            color: str | None = None,
            ax: Axes | None = None
            ) -> None:
        assert orientation in ['vertical', 'horizontal']
        self.orientation = orientation
        
        self.source = source
        if not feature:
            feature = 'feature'
            self.source[feature] = range(len(source[target]))
        self.y = target
        self.x = feature
        
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
        return self._color

    @abstractmethod
    def __call__(self):...


class Scatter(BasePlotter):

    kind = 'scatter'

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '', 
            orientation: Literal['vertical', 'horizontal'] = 'vertical', 
            color: str | None = None,
            marker: str | None = None,
            size: Iterable[int] | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source, target, feature, orientation, color, ax)
        self.size = size
        self.marker = marker
    
    def __call__(self, **kwds):
        kwds = dict(
            c=self.color, marker=self.marker, s=self.size,
            alpha=COLOR.MARKER_ALPHA) | kwds
        self.ax.scatter(
            self.source[self.x], self.source[self.y], **kwds)


class Line(BasePlotter):

    kind = 'line'

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '', 
            orientation: Literal['vertical', 'horizontal'] = 'vertical', 
            color: str | None = None,
            ax: Axes | None = None,
            marker: str | None = None,
            size: Iterable[int] | None = None,
            **kwds) -> None:
        super().__init__(
            source, target, feature, orientation, color, ax)
        self.size = size
        self.marker = marker
    
    def __call__(self, **kwds):
        kwds = dict(
            c=self.color, marker=self.marker, markersize=self.size,
            alpha=COLOR.MARKER_ALPHA) | kwds
        self.ax.plot(self.source[self.x], self.source[self.y], **kwds)


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
            **kwds) -> None:
        self.base = shift
        self.marker = None,
        self.markersize = None
        feature = 'kde'
        sequence, estimation = estimate_kernel_density(
            source[target], height, shift)
        data = pd.DataFrame({target: sequence, feature: estimation})
        super().__init__(
            data, target, feature, orientation, color, ax)
        
    def __call__(self, kw_line: dict, **kw_fill):
        super().plot(**kw_line)
        kw_fill = {'alpha': COLOR.FILL_ALPHA} | kw_fill
        if self.transposed:
            self.ax.fill_betweenx(self.y, self.base, self.x, **kw_fill)
        else:
            self.ax.fill_between(self.x, self.base, self.y, **kw_fill)


__all__ = [
    BasePlotter.__name__,
    Scatter.__name__,
    Line.__name__,
    KDE.__name__
]