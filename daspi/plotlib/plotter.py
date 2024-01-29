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

from .._constants import COLOR
from ..statistics.estimation import estimate_kernel_density


class _Plotter(ABC):

    __slots__ = (
        'source', 'y', 'x', '_color', 'target_axis', 'ax')
    source: Hashable
    y: str
    s: str
    _color: str
    target_axis: str
    fig: Figure
    ax: Axes

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            target_axis : Literal['y', 'x'] = 'y',
            color: str | None = None,
            ax: Axes | None = None,
            ) -> None:
        assert target_axis in ['y', 'x']
        self.target_axis = target_axis
        
        self.source = source
        if not feature:
            feature = 'feature'
            self.source[feature] = range(len(source[target]))

        self.x = self.source[feature]
        self.y = self.source[target]
        if self.transpose: 
            self.x, self.y = self.y, self.x
        
        self.fig, self.ax = plt.subplots(1, 1) if ax is None else ax.figure, ax
        self._color = color
        
    @property
    def transpose(self) -> bool:
        """True if target_axis is 'x'"""
        return self.target_axis == 'x'
    
    @property
    def color(self) -> str | None:
        """Get color of drawn artist"""
        if self._color is None:
            self._color = COLOR.PALETTE[0]
        return self._color

    @abstractmethod
    def __call__(self):...


class Scatter(_Plotter):

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '', 
            target_axis: Literal['y', 'x'] = 'y', 
            color: str | None = None,
            marker: str | None = None,
            size: Iterable[int] | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            target_axis=target_axis, color=color, ax=ax)
        self.size = size
        self.marker = marker
    
    def __call__(self, **kwds):
        kwds = dict(
            c=self.color, marker=self.marker, s=self.size,
            alpha=COLOR.MARKER_ALPHA) | kwds
        self.ax.scatter(self.x, self.y, **kwds)


class Line(_Plotter):

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '', 
            target_axis: Literal['y', 'x'] = 'y', 
            color: str | None = None,
            ax: Axes | None = None,
            marker: str | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            target_axis=target_axis, color=color, ax=ax)
        self.marker = marker
    
    def __call__(self, **kwds):
        kwds = dict(
            c=self.color, marker=self.marker, alpha=COLOR.MARKER_ALPHA) | kwds
        self.ax.plot(self.x, self.y, **kwds)

# TODO: add option to remove density axis
# TODO: add default density label
class KDE(Line):

    __slots__ = ('base')
    base: float

    def __init__(
            self,
            source: Hashable,
            target: str,
            base: float = 0,
            height: float | None = None,
            target_axis: Literal['y', 'x'] = 'x',
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.base = base
        self.marker = None
        feature = 'kde'
        sequence, estimation = estimate_kernel_density(
            source[target], height, base)
        data = pd.DataFrame({target: sequence, feature: estimation})
        super().__init__(
            source=data, target=target, feature=feature,
            target_axis=target_axis, color=color, ax=ax)
        
    def __call__(self, kw_line: dict = {}, **kw_fill):
        super().__call__(**kw_line)
        kw_fill = {'alpha': COLOR.FILL_ALPHA} | kw_fill
        if self.target_axis == 'y':
            self.ax.fill_betweenx(self.y, self.base, self.x, **kw_fill)
        else:
            self.ax.fill_between(self.x, self.base, self.y, **kw_fill)


class Violine(KDE):

    def __init__(
            self,
            source: Hashable,
            target: str,
            base: float = 0,
            width: float = 1,
            target_axis: Literal['y', 'x'] = 'y',
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, base=base, height=width/2, 
            target_axis=target_axis, color=color, ax=ax)

    def __call__(self, **kw_fill):
        kw_fill = {'alpha': COLOR.FILL_ALPHA} | kw_fill
        if self.target_axis == 'y':
            x1 = 2*self.base - self.x
            self.ax.fill_betweenx(self.y, x1, self.x, **kw_fill)
        else:
            y1 = 2*self.base - self.y
            self.ax.fill_between(self.x, y1, self.y, **kw_fill)

__all__ = [
    _Plotter.__name__,
    Scatter.__name__,
    Line.__name__,
    KDE.__name__
]
