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
from typing import Literal
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
        'source', 'target', 'target_on_y', 'axes_facets', 'nrows', 'ncols',
        '_data')
    source: DataFrame
    target: str
    target_on_y: bool
    axes_facets: AxesFacets
    nrows: int
    ncols: int
    _data: DataFrame

    def __init__(
            self, source: DataFrame, target: str, target_on_y: bool = True,
            axes_facets: AxesFacets | None = None, **kwds) -> None:
        self.source = source
        self.target = target
        self.target_on_y = target_on_y
        self.nrows = kwds.pop('nrows', 1)
        self.ncols = kwds.pop('ncols', 1)
        if axes_facets is None:
            self.axes_facets = AxesFacets(self.nrows, self.ncols, **kwds)
        else:
            self.axes_facets = axes_facets
        self._data: DataFrame | None = None

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements"""
        return self.axes_facets.figure
    
    @property
    def axes(self) -> NDArray:
        """Get the created axes"""
        return self.axes_facets.axes
    
    def get_axis_label(self, label: Any, axis: Literal['x', 'y']) -> str:
        """Get axis label according to given axis.
        
        Parameters
        ----------
        label: Any
            If a string is passed, it will be taken. If True, labels of 
            given feature or target name are used taking into account 
            the target_on_y attribute. If False or None, empty string is 
            used, by default ''
        
        Returns
        -------
        label: str
            labels for x or y axis
        """
        assert axis in ['x', 'y']

        get_target = ((axis == 'y' and self.target_on_y) or
                      (axis == 'x' and not self.target_on_y))
        match label:
            case None | False: 
                return ''
            case True: 
                return self.target if get_target else self.feature
            case _:
                return str(label)
    
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
        'feature', 'hue', 'categorical_features', 'coloring',
        'dodging', '_variate_names', '_current_variate', '_last_variate')
    feature: str
    hue: str
    categorical_features: bool
    coloring: HueLabel
    dodging: Dodger
    _current_variate: dict
    _last_variate: dict

    def __init__(self, source: DataFrame, target: str, feature: str = '',
            hue: str = '', dodge: bool = False, 
            categorical_features: bool = False, **kwds) -> None:
        self.categorical_features = categorical_features or dodge
        self.feature = feature
        self.hue = hue
        super().__init__(source=source, target=target, **kwds)
        self.coloring = HueLabel(self.get_categorical_labels(self.hue))
        feature_tick_labels = ()
        if self.categorical_features:
            assert feature in source, 'To be able to use the "categorical_features" attribute, features must be present'
            feature_tick_labels = self.get_categorical_labels(feature)
        dodge_categories = self.coloring.labels if dodge else ()
        self.dodging = Dodger(dodge_categories, feature_tick_labels)
        self._variate_names = (self.hue, )
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
    def legend_handles_labels(self) -> Dict[str, Tuple[tuple]]:
        """Get dictionary of handles and labels
        - keys: titles as str
        - values: handles and labels as tuple of tuples"""
        return {self.hue: self.coloring.handles_labels()}
    
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
        # axis.set_minor_locator(AutoMinorLocator(2))
        # axis.grid(True, which='minor')
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
                ax=self.axes_facets.ax, width=self.dodging.width, **kwds)
            plot()
        return self

    def label(
        self, fig_title: str = '', sub_title: str = '', xlabel: bool | str = '',
        ylabel: bool | str = '', info: bool | str = False) -> Self:
        if self.categorical_features:
            self._correct_feature_ticks_labels_()
        xlabel = self.get_axis_label(xlabel, 'x')
        ylabel = self.get_axis_label(ylabel, 'y')

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            info=info, legends=self.legend_handles_labels)
        label.draw()

        return self


class RelationalChart(SimpleChart):

    __slots__ = ('shape', 'size', 'marking', 'sizing', '_sizes')
    shape: str
    size: str
    marking: ShapeLabel
    sizing: SizeLabel
    _sizes: NDArray

    def __init__(
            self, source: DataFrame, target: str, feature: str,
            hue: str = '', shape: str = '', size: str = '', 
            dodge: bool = False, **kwds):
        self.shape = shape
        self.size = size
        super().__init__(
            source=source, target=target, feature=feature, hue=hue, 
            dodge=dodge, **kwds)
        self.marking = ShapeLabel(self.get_categorical_labels(self.shape))
        if self.size:
            self.sizing = SizeLabel(
                self.source[self.size].min(), self.source[self.size].max())
        else:
            self.sizing = None
        self._sizes: NDArray | None = None
        self._variate_names = (self.hue, self.shape)
        self._reset_variate_()
    
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

    def plot(
            self, plotter: _Plotter, **kwds) -> Self:
        self.target_on_y = kwds.pop('target_on_y', self.target_on_y)
        for data in self:
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=self.target_on_y, color=self.color, 
                ax=self.axes_facets.ax, marker=self.marker, size=self.sizes,
                **kwds)
            plot()
        return self


class JointChart(_Chart):

    __slots__ = ('charts')
    charts: List[RelationalChart]

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
            sharex: bool | str = False,
            sharey: bool | str = False,
            width_ratios: List[float] | None = None,
            height_ratios: List[float] | None = None,
            stretch_figsize: bool = True,
            **kwds) -> None:

        super().__init__(
            source=source, target=target, sharex=sharex, sharey=sharey,
            width_ratios=width_ratios, height_ratios=height_ratios, 
            stretch_figsize=stretch_figsize, nrows=nrows, ncols=ncols, **kwds)

        self.charts = []
        charts_data = dict(
            feature = self.ensure_tuple(feature),
            hue = self.ensure_tuple(hue),
            shape = self.ensure_tuple(shape),
            size = self.ensure_tuple(size),
            dodge = self.ensure_tuple(dodge),
            categorical_features = self.ensure_tuple(categorical_features))
        kw = dict(
            source=self.source, target=self.target, axes_facets=self.axes_facets)
        for values in zip(*charts_data.values()):
            _kw = kw | {k: v for k, v in zip(charts_data.keys(), values)}
            self.charts.append(RelationalChart(**_kw))
    
    @property
    def _idx(self) -> int:
        """Get the index of current ax in flatten axes"""
        for idx, ax in enumerate(self.axes.flat):
            if self.axes_facets.ax == ax:
                return idx
        return 0
    
    @property
    def legend_handles_labels(self) -> dict:
        lh = {}
        for chart in self.charts:
            lh = lh | chart.legend_handles_labels
        return lh
    
    def ensure_tuple(self, attribute: Any) -> tuple:
        """Stellt sicher, dass das angegebene Attribut ein Tupel mit der
        gleichen Nummer wie die Achsen ist. Wird nur ein Wert angegeben,
        wird dieser entsprechend kopiert."""
        amount = self.nrows + self.ncols
        if isinstance(attribute, tuple):
            new_attribute = attribute
        elif isinstance(attribute, list):
            new_attribute = tuple(attribute)
        else:
            new_attribute = tuple(attribute for _ in range(amount))
        assert len(new_attribute) == amount, f'{attribute} does not have enough values, needed {amount}'
        return new_attribute

    def get_axis_label(self, label: Any) -> str | Tuple[str]:
        get_axis_label = lambda l: '' if l in (None, False) else l
        if isinstance(label, (tuple, list)):
            label = tuple(get_axis_label(l) for l in label)
        else:
            label = get_axis_label(label)
        return label
    
    def _data_genenator_(self) -> Generator[Tuple, Self, None]:
        return super()._data_genenator_()
        
    def plot(
            self, plotters_kwds: List[Tuple[_Plotter | None, Dict]],
            hide_none: bool = True) -> Self:
        _axs = iter(self.axes_facets)
        for chart, (plotter, kwds) in zip(self.charts, plotters_kwds):
            ax = next(_axs)
            if plotter is None:
                if hide_none: ax.set_axis_off()
                continue
            chart.plot(plotter, **kwds)
        return self
    
    def label(
            self, fig_title: str = '', sub_title: str = '',
            xlabel: str | Tuple[str] = '', ylabel: str | Tuple[str] = '', 
            row_title: str = '', col_title: str = '',
            info: bool | str = False) -> Self:
        xlabel = self.get_axis_label(xlabel)
        ylabel = self.get_axis_label(ylabel)

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            info=info, row_title=row_title, col_title=col_title,
            legends=self.legend_handles_labels)
        label.draw()
        return self


class MultipleVariateChart(RelationalChart):

    __slots__ = ('col', 'row', 'row_labels', 'col_labels')
    col: str
    row: str
    row_labels: tuple
    col_labels: tuple

    def __init__(
            self, source: DataFrame, target: str, feature: str = '',
            hue: str = '', shape: str = '', size: str = '', col: str = '',
            row: str = '', dodge: bool = False
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
            nrows=nrows, ncols=ncols)
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
        return self
                
    def label(
            self, fig_title: str = '', sub_title: str = '', xlabel: str = '',
            ylabel: str = '', row_title: str = '', col_title: str = '',
            info: bool | str = False) -> Self:
        xlabel = self.get_axis_label(xlabel, 'x')
        ylabel = self.get_axis_label(ylabel, 'y')

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            rows=self.row_labels, cols=self.col_labels,
            info=info, row_title=row_title, col_title=col_title,
            legends=self.legend_handles_labels)
        label.draw()
        return self

__all__ = [
    SimpleChart.__name__,
    RelationalChart.__name__,
    JointChart.__name__,
    MultipleVariateChart.__name__
]