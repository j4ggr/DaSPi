import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Self
from typing import Tuple
from typing import Literal
from typing import Hashable
from typing import Iterable
from typing import Generator
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._constants import COLOR
from .._constants import PLOTTER
from ..statistics.estimation import estimate_kernel_density


class _Plotter(ABC):

    __slots__ = (
        'source', 'y', 'x', '_color', 'target_axis', 'fig', 'ax')
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
        if not self.target_on_y: 
            self.x, self.y = self.y, self.x
        
        self.fig, self.ax = plt.subplots(1, 1) if ax is None else ax.figure, ax
        self._color = color
        
    @property
    def target_on_y(self) -> bool:
        """True if target_axis is 'y'"""
        return self.target_axis == 'y'
    
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
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            target_axis=target_axis, color=color, ax=ax)
    
    def __call__(self, marker=None, **kwds):
        alpha = None if marker is None else COLOR.MARKER_ALPHA
        kwds = dict(c=self.color, marker=marker, alpha=alpha) | kwds
        self.ax.plot(self.x, self.y, **kwds)


class _TransformPlotter(_Plotter):

    __slots__ = ('pos', 'target', 'feature')
    pos: int | float
    target: str
    feature: str
    
    def __init__(
            self,
            source: pd.DataFrame,
            target: str,
            feature: str = '',
            pos: int | float = PLOTTER.TRANSFORMED_DEFAULT_POS,
            target_axis: Literal['y', 'x'] = 'y',
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.pos = pos
        self.target = target
        self.feature = feature
        
        trans_data = pd.DataFrame()
        for _feature, _target in self.feature_grouped(source):
            _data = self.transform(_feature, _target)
            trans_data = pd.concat([trans_data, _data], axis=0)

        super().__init__(
            source=trans_data, target=target, feature=feature,
            target_axis=target_axis, color=color, ax=ax)
    
    def feature_grouped(
            self, source: pd.DataFrame) -> Generator[Tuple, Self, None]:
        if self.feature and self.feature != PLOTTER.TRANSFORMED_FEATURE:
            grouper = source.groupby(self.feature, sort=True)
            for i, (name, group) in enumerate(grouper, start=1):
                pos = name if isinstance(name, (float, int)) else i
                yield pos, group[self.target]
        else:
            self.feature = PLOTTER.TRANSFORMED_FEATURE
            yield self.pos, source[self.target]
    
    @abstractmethod
    def transform(
        self, feature_data: pd.DataFrame, target_data: Any) -> pd.DataFrame:
        ...
    
    @abstractmethod
    def __call__(self): ...
    

class KDE(_TransformPlotter):

    __slots__ = ('height', 'show_density_axis')
    height: float
    show_density_axis: bool

    def __init__(
            self,
            source: Hashable,
            target: str,
            height: float | None = None,
            target_axis: Literal['y', 'x'] = 'x',
            color: str | None = None,
            ax: Axes | None = None,
            show_density_axis: bool = True,
            **kwds) -> None:
        self.height = height
        self.show_density_axis = show_density_axis
        kwds['feature'] = PLOTTER.TRANSFORMED_FEATURE
        super().__init__(
            source=source, target=target, target_axis=target_axis, color=color, 
            ax=ax, **kwds)
        
    def transform(
            self, feature_data: pd.DataFrame, target_data: Any) -> pd.DataFrame:
        sequence, estimation = estimate_kernel_density(
            target_data, height=self.height, base=feature_data)
        return pd.DataFrame({self.target: sequence, self.feature: estimation})
        
    def __call__(self, kw_line: dict = {}, **kw_fill):
        self.ax.plot(self.x, self.y, **kw_line)
        kw_fill = {'alpha': COLOR.FILL_ALPHA} | kw_fill
        if self.target_on_y:
            self.ax.fill_betweenx(self.y, self.pos, self.x, **kw_fill)
        else:
            self.ax.fill_between(self.x, self.pos, self.y, **kw_fill)
        if not self.show_density_axis:
            axis = 'xaxis' if self.target_on_y else 'yaxis'
            spine = 'bottom' if self.target_on_y else 'left'
            getattr(self.ax, axis).set_visible(False)
            self.ax.spines[spine].set_visible(False)


class Violine(_Plotter):

    def __init__(
            self,
            source: pd.DataFrame,
            target: str,
            feature: str = '',
            width: float = 1,
            target_axis: Literal['y', 'x'] = 'y',
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        if feature in source:
            for i, (name, group) in enumerate(source.groupby(feature, sort=True)):
                sequence, estimation = estimate_kernel_density(
                    group[target], width/2, i+1
                )
        _source = pd.DataFrame()

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
    KDE.__name__,
    Violine.__name__
]
