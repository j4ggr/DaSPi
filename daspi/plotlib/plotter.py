import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod

from typing import Self
from typing import List
from typing import Tuple
from typing import Callable
from typing import Literal
from typing import Hashable
from typing import Iterable
from typing import Generator

from numpy.typing import NDArray

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.container import BarContainer

from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.regression.linear_model import RegressionResults

from pandas.api.types import is_scalar
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from .._constants import KW
from .._constants import COLOR
from .._constants import PLOTTER
from .._constants import CATEGORY
from ..statistics.confidence import mean_ci
from ..statistics.confidence import stdev_ci
from ..statistics.confidence import variance_ci
from ..statistics.confidence import prediction_ci
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
    
    def __call__(self, **kwds) -> None:
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
    
    def __call__(self, marker=None, **kwds) -> None:
        alpha = None if marker is None else COLOR.MARKER_ALPHA
        kwds = dict(c=self.color, marker=marker, alpha=alpha) | kwds
        self.ax.plot(self.x, self.y, **kwds)
            

class LinearRegression(_Plotter):

    __slots__ = (
        'model', 'target_fit', 'show_points', 'show_fit_ci', 'show_pred_ci')
    model: RegressionResults
    target_fit: str
    show_points: bool
    show_fit_ci: bool
    show_pred_ci: bool

    def __init__(
            self,
            source: Hashable,
            target: str,
            feature: str,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            show_points: bool = True,
            show_fit_ci: bool = False,
            show_pred_ci: bool = False,
            **kwds) -> None:
        self.target_fit = PLOTTER.FITTED_VALUES_NAME
        self.show_points = show_points
        self.show_fit_ci = show_fit_ci
        self.show_pred_ci = show_pred_ci
        df = source if isinstance(source, DataFrame) else pd.DataFrame(source)
        df = (df
            .sort_values(feature)
            [[feature, target]]
            .dropna(axis=0, how='any')
            .reset_index(drop=True))
        self.model: RegressionResults = sm.OLS(df[target], sm.add_constant(df[feature])).fit()
        df[self.target_fit] = self.model.fittedvalues
        ci_data = pd.DataFrame(
            data = np.array(prediction_ci(self.model)).T, 
            columns = PLOTTER.REGRESSION_CI_NAMES)
        df = pd.concat([df, ci_data], axis=1)
        super().__init__(
            source=df, target=target, feature=feature, target_on_y=target_on_y,
            color=color, ax=ax)
    
    def __call__(
            self, kw_scatter: dict = {}, kw_fit_ci: dict = {},
            kw_pred_ci: dict = {}, **kwds):
        _color = {'color': self.color}
        
        x, y = self.source[self.feature], self.source[self.target_fit]
        if not self.target_on_y: x, y = y, x
        kwds = KW.FIT_LINE | _color | kwds
        self.ax.plot(x, y, **kwds)
        
        if self.show_points:
            kw_scatter = _color | kw_scatter
            self.ax.scatter(self.x, self.y, **kw_scatter)
        
        if self.show_fit_ci:
            kw_fit_ci = KW.FIT_CI | _color | kw_fit_ci
            lower = self.source[PLOTTER.FIT_CI_LOW]
            upper = self.source[PLOTTER.FIT_CI_UPP]
            if self.target_on_y:
                self.ax.fill_between(self.x, lower, upper, **kw_fit_ci)
            else:
                self.ax.fill_betweenx(self.y, lower, upper, **kw_fit_ci)
        
        if self.show_pred_ci:
            kw_pred_ci = KW.PRED_CI | _color | kw_pred_ci
            lower = self.source[PLOTTER.PRED_CI_LOW]
            upper = self.source[PLOTTER.PRED_CI_UPP]
            if self.target_on_y:
                x0, y0 = self.x, lower
                x1, y1 = self.x, upper
            else:
                x0, y0 = lower, self.y
                x1, y1 = upper, self.y
            self.ax.plot(x0, y0, x1, y1, **kw_pred_ci)


class _TransformPlotter(_Plotter):

    __slots__ = ('_f_base')
    _f_base: int | float
    
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            f_base: int | float = PLOTTER.DEFAULT_F_BASE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self._f_base = f_base
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
            for i, (f_value, group) in enumerate(grouper, start=1):
                f_base = f_value if isinstance(f_value, (float, int)) else i
                yield f_base, group[self.target]
        else:
            self.feature = PLOTTER.TRANSFORMED_FEATURE
            yield self._f_base, source[self.target]
    
    @abstractmethod
    def transform(
        self, feature_data: float | int, target_data: Series) -> DataFrame:
        ...
    
    @abstractmethod
    def __call__(self): ...


class Bar(_TransformPlotter):

    __slots__ = ('method', 'kw_method', 'stack', 'width')
    method: str | None
    kw_method: dict
    stack: bool
    width: float

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            stack: bool = True,
            width: float = CATEGORY.FEATURE_SPACE,
            method: str | None = None,
            kw_method: dict = {},
            f_base: int | float = PLOTTER.DEFAULT_F_BASE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.stack = stack
        self.width = width
        self.method = method
        self.kw_method = kw_method
        super().__init__(
            source=source, target=target, feature=feature, f_base=f_base,
            target_on_y=target_on_y, color=color, ax=ax, **kwds)

        if self.method is not None:
            target = f'{self.target} {self.method}'
            self.source = self.source.rename(columns={self.target: target})
            self.target = target

    @property
    def bars(self) -> List[BarContainer]:
        return [c for c in self.ax.containers if isinstance(c, BarContainer)]
    
    @property
    def t_base(self) -> NDArray:
        feature_ticks = self.source[self.feature]
        t_base = np.zeros(len(feature_ticks))
        if not self.stack: 
            return t_base

        for bar in self.bars:
            boxs = [p.get_bbox() for p in bar.patches]
            if self.target_on_y:
                low, upp = map(tuple, zip(*[(b.x0, b.x1) for b in boxs]))
            else:
                low, upp = map(tuple, zip(*[(b.y0, b.y1) for b in boxs]))
            if (all(np.greater(feature_ticks, low))
                and all(np.less(feature_ticks, upp))
                and any(np.greater(bar.datavalues, t_base))):
                t_base = bar.datavalues
        return t_base
    
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        if self.method is not None:
            t_value = getattr(target_data, self.method)(**self.kw_method)
            assert is_scalar(t_value), (
                f'{self.method} does not return a scalar')
            t_value = [t_value]
        else:
            t_value = target_data
            assert len(t_value) <= 1, (
                'Each feature level must contain only one target value')
        
        data = pd.DataFrame({
            self.target: t_value,
            self.feature: [feature_data]})
        return data

    def __call__(self, **kwds) -> None:
        if self.target_on_y:
            self.ax.bar(
                self.x, self.y, width=self.width, bottom=self.t_base, **kwds)
        else:
            self.ax.barh(
                self.y, self.x, height=self.width, left=self.t_base, **kwds)


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
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        data = pd.DataFrame({
            self.target: target_data,
            self.feature: self.jitter(feature_data, target_data.size)})
        return data

    def __call__(self, **kwds) -> None:
        kwds = dict(color=self.color) | kwds
        self.ax.scatter(self.x, self.y, **kwds)


class GaussianKDE(_TransformPlotter):

    __slots__ = ('_height', 'show_density_axis', 'base_on_zero')
    _height: float
    show_density_axis: bool
    base_on_zero: bool

    def __init__(
            self,
            source: Hashable,
            target: str,
            height: float | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            show_density_axis: bool = True,
            base_on_zero: bool = True,
            **kwds) -> None:
        self._height = height
        self.show_density_axis = show_density_axis
        self.base_on_zero = base_on_zero
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
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        sequence, estimation = estimate_kernel_density(
            target_data, height=self.height, base=feature_data)
        data = pd.DataFrame({
            self.target: sequence,
            self.feature: estimation,
            PLOTTER.F_BASE_NAME: feature_data * np.ones(len(sequence))})
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
            self.ax.fill_betweenx(self.y, self._f_base, self.x, **kw_fill)
        else:
            self.ax.fill_between(self.x, self._f_base, self.y, **kw_fill)
        if not self.show_density_axis:
            self.hide_density_axis()
        if self.base_on_zero:
            xy = 'x' if self.target_on_y else 'y'
            getattr(self.ax, f'set_{xy}margin')(0)


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
            show_density_axis=True, base_on_zero=False, **kwds)

    def __call__(self, **kwds) -> None:
        kwds = dict(color=self.color, alpha=COLOR.FILL_ALPHA) | kwds
        for f_base, group in self.source.groupby(PLOTTER.F_BASE_NAME):
            estim_upp = group[self.feature]
            estim_low = 2*f_base - estim_upp
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
            show_density_axis=True, base_on_zero=False, **kwds)

    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        f_base = feature_data + PLOTTER.RIDGE_SHIFT
        sequence, estimation = estimate_kernel_density(
            target_data, stretch=self.stretch, base=f_base)
        data = pd.DataFrame({
            self.target: sequence,
            self.feature: estimation,
            PLOTTER.F_BASE_NAME: f_base * np.ones(len(sequence))})
        return data
    
    def __call__(self, **kwds) -> None:
        kwds = dict(color=self.color, alpha=COLOR.FILL_ALPHA) | kwds
        for f_base, group in self.source.groupby(PLOTTER.F_BASE_NAME):
            estimation = group[self.feature]
            sequence = group[self.target]
            if self.target_on_y:
                self.ax.plot(estimation, sequence, c=COLOR.WHITE_TRANSPARENT)
                self.ax.fill_betweenx(sequence, f_base, estimation, **kwds)
            else:
                self.ax.plot(sequence, estimation, c=COLOR.WHITE_TRANSPARENT)
                self.ax.fill_between(sequence, f_base, estimation, **kwds)


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
            self, feature_data: float | int, target_data: Series) -> DataFrame:
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
            self, feature_data: float | int, target_data: Series) -> DataFrame:
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
    LinearRegression.__name__,
    Bar.__name__,
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
