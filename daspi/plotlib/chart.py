import numpy as np
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Self
from typing import Dict
from typing import List
from typing import Tuple
from typing import Generator
from pathlib import Path
from numpy.typing import NDArray
from pandas.core.frame import DataFrame

from .utils import Dodger
from .utils import HueLabel
from .utils import SizeLabel
from .utils import ShapeLabel
from .facets import LabelFacets
from .facets import AxesFacets
from .plotter import _Plotter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from .._constants import KW
from .._constants import COLOR


class _Chart(ABC):

    __slots__ = (
        'source', 'target', 'feature', 'target_on_y', 'axes_facets', 'nrows',
        'ncols', '_data', '_xlabel', '_ylabel', '_plots')
    source: DataFrame
    target: str
    feature: str
    target_on_y: bool
    axes_facets: AxesFacets
    nrows: int
    ncols: int
    _data: DataFrame
    _xlabel: str
    _ylabel: str
    _plots: List[_Plotter]

    def __init__(
            self, source: DataFrame, target: str | Tuple[str], 
            feature: str | Tuple[str]= '', target_on_y: bool = True, 
            axes_facets: AxesFacets | None = None, **kwds) -> None:
        self.source = source
        self.target = target
        self.feature = feature
        self.nrows = kwds.pop('nrows', 1)
        self.ncols = kwds.pop('ncols', 1)
        if axes_facets is None:
            self.axes_facets = AxesFacets(self.nrows, self.ncols, **kwds)
        else:
            self.axes_facets = axes_facets
        self.target_on_y = target_on_y
        for ax in self.axes.flat:
            getattr(ax, f'set_{"x" if self.target_on_y else "y"}margin')(0)
        self._data: DataFrame | None = None
        self._xlabel = ''
        self._ylabel = ''
        self._plots = []

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements"""
        return self.axes_facets.figure
    
    @property
    def axes(self) -> NDArray:
        """Get the created axes"""
        return self.axes_facets.axes
    
    @property
    def n_axes(self) -> int:
        """Get amount of axes"""
        return self.ncols * self.nrows

    @property
    def xlabel(self) -> str:
        """Get label for x axis, set with `set_axis_label` method"""
        return self._xlabel
    
    @property
    def ylabel(self) -> str:
        """Get label for y axis, set with `set_axis_label` method"""
        return self._ylabel
    
    @property
    def plots(self) -> List[_Plotter]:
        """Get plotter objects used in `plot` method"""
        return self._plots
    
    def set_axis_label(
            self, label: Any, is_target: bool) -> str:
        """Set axis label according to given kind of label, taking into
        account the `target_on_y` attribute.
        
        Parameters
        ----------
        label : Any
            If a string is passed, it will be taken. If True, labels of 
            given feature or target name are used taking into account 
            the target_on_y attribute. If False or None, empty string is 
            used, by default ''
        is_target : Bool
            Set True if the label is for the target axis
        """

        match label:
            case None | False: 
                _label = ''
            case True: 
                _label = self.target if is_target else self.feature
            case _:
                _label = str(label)
        
        if is_target == self.target_on_y:
            self._ylabel = _label
        else:
            self._xlabel = _label
        
    @abstractmethod
    def _data_genenator_(self) -> Generator[Tuple, Self, None]:
        """Implement the data generator and add the currently yielded 
        data to self._data so that it can be used internally."""
    
    def __iter__(self) -> Generator[Tuple, Self, None]:
        return self._data_genenator_()
        
    def __next__(self) -> Axes:
        return next(self)
    
    @abstractmethod
    def plot(self, plotter: _Plotter): ...

    @abstractmethod
    def label(self, **labels): ...
    
    def save(self, file_name: str | Path, **kwds) -> Self:
        kw = KW.SAVE_CHART | kwds
        self.figure.savefig(file_name, **kw)
        return self

    def close(self):
        """"Close figure"""
        plt.close(self.figure)


class SimpleChart(_Chart):

    __slots__ = (
        'hue', 'shape', 'size', 'marking', 'sizing', '_sizes', 
        'categorical_features', 'coloring', 'dodging', '_variate_names',
        '_current_variate', '_last_variate', )
    hue: str
    shape: str
    size: str
    marking: ShapeLabel
    sizing: SizeLabel
    _sizes: NDArray
    categorical_features: bool
    coloring: HueLabel
    dodging: Dodger
    _current_variate: dict
    _last_variate: dict

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            hue: str = '',
            dodge: bool = False,
            shape: str = '',
            size: str = '', 
            categorical_features: bool = False,
            **kwds) -> None:
        self.categorical_features = categorical_features or dodge
        self.hue = hue
        self.shape = shape
        self.size = size
        super().__init__(source=source, target=target, feature=feature, **kwds)
        self.coloring = HueLabel(self.get_categorical_labels(self.hue))
        feature_tick_labels = ()
        if self.categorical_features:
            assert feature in source, (
                'categorical_features is True, but features is not present')
            feature_tick_labels = self.get_categorical_labels(feature)
        dodge_categories = self.coloring.labels if dodge else ()
        self.dodging = Dodger(dodge_categories, feature_tick_labels)
        self.marking = ShapeLabel(self.get_categorical_labels(self.shape))
        if self.size:
            self.sizing = SizeLabel(
                self.source[self.size].min(), self.source[self.size].max())
        else:
            self.sizing = None
        self._sizes: NDArray | None = None
        self._variate_names = (self.hue, self.shape)
        self._current_variate = {}
        self._last_variate = {}
        self._reset_variate_()
    
    @property
    def variate_names(self) -> List[str]:
        """Get names of all set variates"""
        return [v for v in self._variate_names if v]
    
    @property
    def color(self) -> str:
        """Get color for current variate"""
        hue_variate = self._current_variate.get(self.hue, None)
        return self.coloring[hue_variate]
    
    @property
    def marker(self) -> str:
        """Get marker for current variate"""
        marker_variate = self._current_variate.get(self.shape, None)
        return self.marking[marker_variate]

    @property
    def sizes(self) -> NDArray | None:
        """Get sizes for current variate, is set in grouped data 
        generator."""
        if not self.size: return None
        return self.sizing(self._data[self.size])
    
    @property
    def legend_handles_labels(self) -> Dict[str, Tuple[tuple]]:
        """Get dictionary of handles and labels
        - keys: titles as str
        - values: handles and labels as tuple of tuples"""
        handlers = (self.coloring, self.marking, self.sizing)
        titles = (self.hue, self.shape, self.size)
        return {t: h.handles_labels() for t, h in zip(titles, handlers) if t}
    
    def get_categorical_labels(self, colname: str) -> Tuple:
        """Get sorted unique elements of given column name is in source"""
        if not colname: return ()
        return tuple(sorted(np.unique(self.source[colname])))
    
    def _reset_variate_(self):
        """Set values to None for current and last variate"""
        self._current_variate = {k: None for k in self.variate_names}
        self._last_variate = {k: None for k in self.variate_names}

    def update_variate(self, combination: Any) -> None:
        """Update current variate by given combination coming from
        pandas DataFrame groupby function."""
        if not isinstance(combination, tuple): combination = (combination, )
        self._last_variate = deepcopy(self._current_variate)
        for key, name in zip(self.variate_names, combination):
            self._current_variate[key] = name
    
    def dodge(self) -> None:
        """Converts the feature data to tick positions, taking dodging 
        into account."""
        if not self.dodging: return
        hue_variate = self._current_variate.get(self.hue, None)
        self._data[self.feature] = self.dodging(
            self._data[self.feature], hue_variate)
        
    def _correct_feature_ticks_labels_(self) -> None:
        """Correct feature ticks and labels if according to dodging labels"""
        xy = 'x' if self.target_on_y else 'y'
        _ticks = self.dodging.ticks
        settings = {
            f'{xy}ticks': self.dodging.ticks,
            f'{xy}ticklabels': self.dodging.tick_lables,
            f'{xy}lim': (np.min(_ticks) - 0.5, np.max(_ticks) + 0.5)}
        self.axes_facets.ax.set(**settings)
        self.axes_facets.ax.tick_params(which='minor', color=COLOR.TRANSPARENT)
        
        axis = getattr(self.axes_facets.ax, f'{xy}axis')
        axis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(True, which='minor')
        axis.grid(False, which='major')

    def _data_genenator_(self) -> Generator[Tuple, Self, None]:
        if self.variate_names:
            for combination, data in self.source.groupby(self.variate_names):
                self._data = data
                self.update_variate(combination)
                self.dodge()
                yield self._data
        else:
            self._data = self.source
            yield self._data
        self._reset_variate_()

    def plot(
            self, plotter: _Plotter, **kwds) -> Self:
        self.target_on_y = kwds.pop('target_on_y', self.target_on_y)
        for data in self:
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=self.target_on_y, color=self.color, 
                ax=self.axes_facets.ax, marker=self.marker, size=self.sizes,
                width=self.dodging.width, **kwds)
            plot()
            self._plots.append(plot)
        return self

    def label(
        self, fig_title: str = '', sub_title: str = '',
        feature_label: bool | str = '', target_label: bool | str = '',
        info: bool | str = False) -> Self:
        if self.categorical_features:
            self._correct_feature_ticks_labels_()
        self.set_axis_label(feature_label, is_target=False)
        self.set_axis_label(target_label, is_target=True)

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=self._xlabel, ylabel=self._ylabel,
            info=info, legends=self.legend_handles_labels)
        label.draw()

        return self


class JointChart(_Chart):

    __slots__ = ('charts', '_target', '_feature')
    charts: List[SimpleChart]
    hue: str | Tuple[str]
    shape: str | Tuple[str]
    size: str | Tuple[str]
    dodge: bool | Tuple[bool]

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str | Tuple[str],
            nrows: int,
            ncols: int,
            hue: str | Tuple[str] = '',
            shape: str | Tuple[str] = '',
            size: str | Tuple[str] = '',
            dodge: bool | Tuple[bool] = False,
            categorical_features: bool | Tuple[str] = False,
            target_on_y: bool | List[bool] = True,
            sharex: bool | str = False,
            sharey: bool | str = False,
            width_ratios: List[float] | None = None,
            height_ratios: List[float] | None = None,
            stretch_figsize: bool = True,
            **kwds) -> None:

        self.charts = []
        self.ncols = ncols
        self.nrows = nrows
        target = self.ensure_tuple(target)
        feature = self.ensure_tuple(feature)
        super().__init__(
            source=source, target=target, feature=feature, 
            sharex=sharex, sharey=sharey, width_ratios=width_ratios,
            height_ratios=height_ratios, stretch_figsize=stretch_figsize,
            nrows=nrows, ncols=ncols, **kwds)
        
        charts_data = dict(
            target = self._target,
            feature = self._feature,
            hue = self.ensure_tuple(hue),
            shape = self.ensure_tuple(shape),
            size = self.ensure_tuple(size),
            dodge = self.ensure_tuple(dodge),
            categorical_features = self.ensure_tuple(categorical_features),
            target_on_y = self.ensure_tuple(target_on_y))
        kw = dict(
            source=self.source, axes_facets=self.axes_facets)
        for values in zip(*charts_data.values()):
            _kw = kw | {k: v for k, v in zip(charts_data.keys(), values)}
            self.charts.append(SimpleChart(**_kw))
    
    @property
    def _idx(self) -> int:
        """Get the index of current ax in flatten axes"""
        for idx, ax in enumerate(self.axes.flat):
            if self.axes_facets.ax == ax:
                return idx
        return 0
    
    @property
    def feature(self) -> str:
        return self._feature[self._idx]
    @feature.setter
    def feature(self, feature: str | Tuple[str]) -> None:
        self._feature: Tuple[str] = self.ensure_tuple(feature)
    
    @property
    def target(self) -> str:
        return self._target[self._idx]
    @target.setter
    def target(self, target: str | Tuple[str]) -> None:
        self._target: Tuple[str] = self.ensure_tuple(target)
    
    @property
    def target_on_y(self) -> List[bool]:
        """Get target_on_y of each sub chart as list

        Set target_on_y for all sub charts. if only one value is given, 
        it is adopted for all sub charts. If list or tuple, the values 
        are assigned to the subcharts in order"""
        return [c.target_on_y for c in self.charts]
    @target_on_y.setter
    def target_on_y(self, target_on_y: bool | List[bool] | Tuple[bool]) -> None:
        for toy, chart in zip(self.ensure_tuple(target_on_y), self.charts):
            chart.target_on_y = toy

    @property
    def same_target_on_y(self) -> bool:
        """True if all target_on_y have the same boolean value."""
        return all(toy == self.target_on_y[0] for toy in self.target_on_y)
    
    @property
    def legend_handles_labels(self) -> Dict:
        legend_hl = {}
        for chart in self.charts:
            legend_hl = legend_hl | chart.legend_handles_labels
        return legend_hl
    
    @property
    def xlabel(self) -> str | Tuple[str]:
        if not self._xlabel:
            self._xlabel = tuple((c.xlabel for c in self.charts))
        return self._xlabel
    
    @property
    def ylabel(self) -> str | Tuple[str]:
        if not self._ylabel: 
            self._ylabel = tuple((c.ylabel for c in self.charts))
        return self._ylabel
    
    def itercharts(self):
        """Iter over charts simultaneosly iters over axes of 
        `axes_facets`. That ensures that the current Axes to which the 
        current chart belongs is set."""
        for _, chart in zip(self.axes_facets, self.charts):
            yield chart
    
    def ensure_tuple(self, attribute: Any) -> Tuple:
        """Ensures that the specified attribute is a tuple with the same
        length as the axes. If only one value is specified, it will be
        copied accordingly."""
        if isinstance(attribute, tuple):
            new_attribute = attribute
        elif isinstance(attribute, list):
            new_attribute = tuple(attribute)
        else:
            new_attribute = tuple(attribute for _ in range(self.n_axes))
        assert len(new_attribute) == self.n_axes, (
            f'{attribute} does not have enough values, needed {self.n_axes}')
        return new_attribute

    def set_axis_label(
            self, label: Any, is_target: bool) -> None:
        if label and isinstance(label, str):
            assert self.same_target_on_y, (
                'For a single label, all axes must have the same orientation')
            if is_target == all(self.target_on_y):
                self._ylabel = label
            else:
                self._xlabel = label
        else:
            for _label, chart in zip(self.ensure_tuple(label), self.charts):
                chart.set_axis_label(_label, is_target=is_target)
    
    def _data_genenator_(self) -> Generator[Tuple, Self, None]:
        return super()._data_genenator_()
        
    def plot(
            self, plotters_kwds: List[Tuple[_Plotter | None, Dict]],
            hide_none: bool = True) -> Self:
        _axs = iter(self.axes_facets)
        for chart, (plotter, kwds) in zip(self.itercharts(), plotters_kwds):
            ax = next(_axs)
            if plotter is None:
                if hide_none: ax.set_axis_off()
                continue
            chart.plot(plotter, **kwds)
            self._plots.extend(chart.plots)
        return self
    
    def label(
            self, fig_title: str = '', sub_title: str = '',
            feature_label: str | bool | Tuple = '', 
            target_label: str | bool | Tuple = '', 
            row_title: str = '', col_title: str = '', info: bool | str = False
            ) -> Self:
        for chart in self.itercharts():
            if not chart.categorical_features: continue
            chart._correct_feature_ticks_labels_()
        self._xlabel = ''
        self._ylabel = ''
        self.set_axis_label(feature_label, is_target=False)
        self.set_axis_label(target_label, is_target=True)

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=self.xlabel, ylabel=self.ylabel,
            info=info, row_title=row_title, col_title=col_title,
            legends=self.legend_handles_labels)
        label.draw()
        return self


class MultipleVariateChart(SimpleChart):

    __slots__ = ('col', 'row', 'row_labels', 'col_labels')
    col: str
    row: str
    row_labels: tuple
    col_labels: tuple

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            hue: str = '',
            shape: str = '',
            size: str = '',
            col: str = '',
            row: str = '',
            dodge: bool = False,
            stretch_figsize: bool = True,
            ) -> None:
        self.target_on_y = True
        self.source = source
        self.col = col
        self.row = row
        self.row_labels = self.get_categorical_labels(self.row)
        self.col_labels = self.get_categorical_labels(self.col)
        nrows = max([1, len(self.row_labels)])
        ncols = max([1, len(self.col_labels)])
        super().__init__(
            source=self.source, target=target, feature=feature, hue=hue, 
            shape=shape, size=size, dodge=dodge, sharex=True, sharey=True,
            nrows=nrows, ncols=ncols, stretch_figsize=stretch_figsize)
        self._variate_names = (self.row, self.col, self.hue, self.shape)
        self._reset_variate_()
    
    @property
    def row_or_col_changed(self) -> bool:
        """Check whether the current variate belongs to a new row 
        or column relative to the last variate"""
        for key, old in self._last_variate.items():
            new = self._current_variate[key]
            if old != new and key in (self.row, self.col):
                return True
        return False

    def plot(self, plotter: _Plotter) -> Self:
        ax = None
        _ax = iter(self.axes_facets)
        for data in self:
            if self.row_or_col_changed or ax is None: ax = next(_ax)
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=self.target_on_y, color=self.color, ax=ax, 
                marker=self.marker, size=self.sizes)
            plot()
            self._plots.append(plot)
        return self
                
    def label(
            self, feature_label: str, target_label: str,
            fig_title: str = '', sub_title: str = '', row_title: str = '',
            col_title: str = '', info: bool | str = False
            ) -> Self:
        self.set_axis_label(feature_label, is_target=False)
        self.set_axis_label(target_label, is_target=True)

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=self.xlabel, ylabel=self.ylabel,
            rows=self.row_labels, cols=self.col_labels,
            info=info, row_title=row_title, col_title=col_title,
            legends=self.legend_handles_labels)
        label.draw()
        return self

__all__ = [
    SimpleChart.__name__,
    SimpleChart.__name__,
    JointChart.__name__,
    MultipleVariateChart.__name__
]