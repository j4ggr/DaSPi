import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Self
from typing import Tuple
from typing import Callable
from typing import Literal
from typing import Hashable
from typing import Iterable
from typing import Generator
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._constants import KW
from .._constants import COLOR
from .._constants import PLOTTER
from .._constants import CATEGORY
from ..statistics.confidence import mean_ci
from ..statistics.confidence import stdev_ci
from ..statistics.confidence import variance_ci
from ..statistics.estimation import estimate_kernel_density


class _Plotter(ABC):

    __slots__ = (
        'source', 'target', 'feature', '_color', 'target_on_y', 'fig', 'ax')
    source: Hashable
    target: str
    feature: str
    _color: str
    target_on_y: bool
    fig: Figure
    ax: Axes

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            target_on_y : bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            ) -> None:
        self.target_on_y = target_on_y
        self.source = source
        if not feature:
            feature = PLOTTER.FEATURE
            self.source[feature] = np.arange(len(source[target]))
        self.feature = feature
        self.target = target
        
        self.fig, self.ax = plt.subplots(1, 1) if ax is None else ax.figure, ax
        self._color = color
    
    @property
    def x(self):
        """Get values used for x axis"""
        name = self.feature if self.target_on_y else self.target
        return self.source[name]
    
    @property
    def y(self):
        """Get values used for y axis"""
        name = self.target if self.target_on_y else self.feature
        return self.source[name]
    
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
            target_on_y: bool = True, 
            color: str | None = None,
            marker: str | None = None,
            size: Iterable[int] | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
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
            target_on_y: bool = True, 
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
    
    def __call__(self, marker=None, **kwds):
        alpha = None if marker is None else COLOR.MARKER_ALPHA
        kwds = dict(c=self.color, marker=marker, alpha=alpha) | kwds
        self.ax.plot(self.x, self.y, **kwds)


class Bar(_Plotter):

    __slots__ = ('base', 'width')
    base: str
    width: float

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str,
            base: Any = None,
            width: float = CATEGORY.FEATURE_SPACE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None) -> None:
        source = pd.DataFrame(source).copy()
        if base is None: base = 0
        if isinstance(base, str):
            self.base = base
        else:
            self.base = PLOTTER.BASE
            source[self.base] = base
        self.width = width
            
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)

    def __call__(self, **kwds):
        if self.target_on_y:
            self.ax.bar(
                self.x, self.y, width=self.width, bottom=self.source[self.base],
                **kwds)
        else:
            self.ax.barh(
                self.y, self.x, height=self.width, left=self.source[self.base],
                **kwds)


class _TransformPlotter(_Plotter):

    __slots__ = ('_base')
    _base: int | float
    
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            base: int | float = PLOTTER.DEFAULT_BASE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self._base = base
        self.target = target
        self.feature = feature
        
        trans_data = pd.DataFrame()
        for _feature, _target in self.feature_grouped(source):
            _data = self.transform(_feature, _target)
            trans_data = pd.concat([trans_data, _data], axis=0)

        super().__init__(
            source=trans_data, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
    
    def feature_grouped(
            self, source: DataFrame) -> Generator[Tuple, Self, None]:
        if self.feature and self.feature != PLOTTER.TRANSFORMED_FEATURE:
            grouper = source.groupby(self.feature, sort=True)
            for i, (name, group) in enumerate(grouper, start=1):
                base = name if isinstance(name, (float, int)) else i
                yield base, group[self.target]
        else:
            self.feature = PLOTTER.TRANSFORMED_FEATURE
            yield self._base, source[self.target]
    
    @abstractmethod
    def transform(
        self, feature_data: float | int, target_data: Series
        ) -> pd.DataFrame:
        ...
    
    @abstractmethod
    def __call__(self): ...


class Jitter(_TransformPlotter):

    __slots__ = ('width')
    width: float

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            width: float = CATEGORY.FEATURE_SPACE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.width = width
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax, **kwds)
        
    def jitter(self, loc: float, size: int) -> NDArray:
        """Generates normally distributed jitter values. The standard 
        deviation is selected so that +- 6 sigma corresponds to the 
        permissible width. To ensure the width, values that lie outside 
        this range are restricted to the limits.
        
        Parameters
        ----------
        loc : float
            Center position (feature axis) of the jitted values.
        size : int
            Amount of valaues to generate
        
        Returns
        -------
        jitter : 1D array
            Normally distributed values, but not wider than the given 
            width
        """
        jiiter = np.clip(
            np.random.normal(loc=loc, scale=self.width/6, size=size),
            a_min = loc - self.width/2,
            a_max = loc + self.width/2)
        return jiiter
        
    def transform(
            self, feature_data: float | int, target_data: Series
            ) -> pd.DataFrame:
        data = pd.DataFrame({
            self.target: target_data,
            self.feature: self.jitter(feature_data, target_data.size)})
        return data

    def __call__(self, **kwds) -> None:
        kwds = dict(color=self.color) | kwds
        self.ax.scatter(self.x, self.y, **kwds)


class GaussianKDE(_TransformPlotter):

    __slots__ = ('_height', 'show_density_axis')
    _height: float
    show_density_axis: bool

    def __init__(
            self,
            source: Hashable,
            target: str,
            height: float | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            show_density_axis: bool = True,
            **kwds) -> None:
        self._height = height
        self.show_density_axis = show_density_axis
        feature = PLOTTER.TRANSFORMED_FEATURE
        _feature = kwds.pop('feature', '')
        if type(self) != GaussianKDE and _feature:
            feature = _feature
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax, **kwds)
        
    @property
    def height(self) -> float:
        """Height of kde curve at its maximum."""
        return self._height
        
    def transform(
            self, feature_data: float | int, target_data: Series
            ) -> pd.DataFrame:
        sequence, estimation = estimate_kernel_density(
            target_data, height=self.height, base=feature_data)
        data = pd.DataFrame({
            self.target: sequence,
            self.feature: estimation,
            PLOTTER.BASE: feature_data * np.ones(len(sequence))})
        return data
    
    def hide_density_axis(self) -> None:
        axis = 'xaxis' if self.target_on_y else 'yaxis'
        spine = 'bottom' if self.target_on_y else 'left'
        getattr(self.ax, axis).set_visible(False)
        self.ax.spines[spine].set_visible(False)
        
    def __call__(self, kw_line: dict = {}, **kw_fill) -> None:
        self.ax.plot(self.x, self.y, **kw_line)
        kw_fill = dict(alpha=COLOR.FILL_ALPHA) | kw_fill
        if self.target_on_y:
            self.ax.fill_betweenx(self.y, self._base, self.x, **kw_fill)
        else:
            self.ax.fill_between(self.x, self._base, self.y, **kw_fill)
        if not self.show_density_axis:
            self.hide_density_axis()


class Violine(GaussianKDE):

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            width: float = CATEGORY.FEATURE_SPACE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self._height = width/2
        super().__init__(
            source=source, target=target, feature=feature, height=self.height,
            target_on_y=target_on_y, color=color, ax=ax,
            show_density_axis=True, **kwds)

    def __call__(self, **kwds) -> None:
        kwds = dict(color=self.color, alpha=COLOR.FILL_ALPHA) | kwds
        for base, group in self.source.groupby(PLOTTER.BASE):
            estim_upp = group[self.feature]
            estim_low = 2*base - estim_upp
            sequence = group[self.target]
            if self.target_on_y:
                self.ax.fill_betweenx(sequence, estim_low, estim_upp, **kwds)
            else:
                self.ax.fill_between(sequence, estim_low, estim_upp, **kwds)


class Ridge(GaussianKDE):

    __slots__ = ('stretch')
    stretch: float

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        _, estim = estimate_kernel_density(source[target])
        self.stretch = 1/np.max(estim)
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax,
            show_density_axis=True, **kwds)

    def transform(
            self, feature_data: float | int, target_data: Series
            ) -> pd.DataFrame:
        base = feature_data+PLOTTER.RIDGE_SHIFT
        sequence, estimation = estimate_kernel_density(
            target_data, stretch=self.stretch, base=base)
        data = pd.DataFrame({
            self.target: sequence,
            self.feature: estimation,
            PLOTTER.BASE: base * np.ones(len(sequence))})
        return data
    
    def __call__(self, **kwds) -> None:
        kwds = dict(color=self.color, alpha=COLOR.FILL_ALPHA) | kwds
        for base, group in self.source.groupby(PLOTTER.BASE):
            estimation = group[self.feature]
            sequence = group[self.target]
            if self.target_on_y:
                self.ax.plot(estimation, sequence, c=COLOR.WHITE_TRANSPARENT)
                self.ax.fill_betweenx(sequence, base, estimation, **kwds)
            else:
                self.ax.plot(sequence, estimation, c=COLOR.WHITE_TRANSPARENT)
                self.ax.fill_between(sequence, base, estimation, **kwds)


class Errorbar(_TransformPlotter):
    __slots__ = ('lower', 'upper', 'show_points', 'sort')
    lower: str
    upper: str
    show_points: bool

    def __init__(
            self,
            source: Hashable,
            target: str,
            lower: str,
            upper: str,
            feature: str = '',
            show_points: bool = True,
            target_on_y: bool = True,
            sort: Literal['feature', 'target', None] = None,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        """
        sort : {'feature', 'target', None}, optional
            sort error bars according to target center points or feature
            values. If 'feature, feature values are sorted 
            descending if target is on y axis (lowest on top), otherwise 
            they will be sorted ascending (lowest on left). If 'target' 
            centers are sorted ascending if target is on y axis 
            (smallest on left), otherweise centers will be sorted 
            descending (smallest on top), by default False"""
        self.lower = lower
        self.upper = upper
        self.show_points = show_points
        if not feature in source:
            feature = PLOTTER.FEATURE
            source[feature] = np.arange(len(source[target]))

        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
        
        if sort == 'target':
            self.source = self.source.sort_values(
                self.target, ascending=self.target_on_y)
        elif sort == 'feature':
            self.source = self.source.sort_values(
                self.feature, ascending=not self.target_on_y)
        
    def transform(
            self, feature_data: float | int, target_data: Series
            ) -> pd.DataFrame:
        data = pd.DataFrame({
            self.target: target_data,
            self.feature: [feature_data]})
        return data
    
    @property
    def err(self) -> NDArray:
        """Get separated error lengths as 2D array. 
        First row contains the lower errors, the second row contains the 
        upper errors."""
        err = np.array([
            self.source[self.target] - self.source[self.lower],
            self.source[self.upper] - self.source[self.target]])
        return err
    
    def __call__(self, kw_points: dict = {}, **kwds):
        if self.show_points:
            kw_points = dict(color=self.color) | kw_points
            self.ax.scatter(self.x, self.y, **kw_points)
        kwds = KW.ERROR_BAR | kwds
        if self.target_on_y:
            self.ax.errorbar(self.x, self.y, yerr=self.err, **kwds)
        else:
            self.ax.errorbar(self.x, self.y, xerr=self.err, **kwds)


class StandardErrorMean(Errorbar):

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            show_points: bool = True,
            target_on_y: bool = True,
            sort: Literal['feature', 'target'] | None = None,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        
        super().__init__(
            source=source, target=target, lower=PLOTTER.ERR_LOW,
            upper=PLOTTER.ERR_UPP, feature=feature, show_points=show_points,
            target_on_y=target_on_y, sort=sort, color=color, ax=ax)

    def transform(
            self, feature_data: float | int, target_data: Series
            ) -> pd.DataFrame:
        center = target_data.mean()
        err = target_data.sem()
        data = pd.DataFrame({
            self.target: [center],
            self.feature: [feature_data],
            self.lower: [center - err],
            self.upper: [center + err]})
        return data


class DistinctionTest(Errorbar):

    __slots__ = ('confidence_level', 'ci_func', 'n_groups')
    confidence_level: float
    ci_func: Callable
    n_groups: int

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            show_points: bool = True,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            ci_func: Callable = mean_ci,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.confidence_level = confidence_level
        self.ci_func = ci_func
        self.n_groups = pd.Series(source[feature]).nunique() if feature else 1
        
        super().__init__(
            source=source, target=target, lower=PLOTTER.ERR_LOW,
            upper=PLOTTER.ERR_UPP, feature=feature, show_points=show_points,
            target_on_y=target_on_y, sort='target', color=color, ax=ax)
    
    def transform(
            self, feature_data: float | int, target_data: Series
            ) -> pd.DataFrame:
        center, lower, upper = self.ci_func(
            target_data, self.confidence_level, self.n_groups)
        data = pd.DataFrame({
            self.target: [center],
            self.feature: [feature_data],
            self.lower: lower,
            self.upper: upper})
        return data


class MeanTest(DistinctionTest):

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            show_points: bool = True,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            show_points=show_points, target_on_y=target_on_y,
            confidence_level=confidence_level, ci_func=mean_ci, color=color,
            ax=ax)


class VariationTest(DistinctionTest):

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str = '',
            show_points: bool = True,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            kind: Literal['stdev', 'variance'] = 'stdev',
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        ci_func = stdev_ci if kind == 'stdev' else variance_ci
        super().__init__(
            source=source, target=target, feature=feature,
            show_points=show_points, target_on_y=target_on_y,
            confidence_level=confidence_level, ci_func=ci_func, color=color,
            ax=ax)


__all__ = [
    _Plotter.__name__,
    Scatter.__name__,
    Line.__name__,
    Jitter.__name__,
    GaussianKDE.__name__,
    Violine.__name__,
    Ridge.__name__,
    Errorbar.__name__,
    StandardErrorMean.__name__,
    DistinctionTest.__name__,
    MeanTest.__name__,
    VariationTest.__name__,
    ]
