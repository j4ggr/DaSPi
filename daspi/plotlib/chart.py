import numpy as np
import pandas as pd
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
from .._constants import PLOTTER


class _Chart(ABC):

    target_on_y: bool = True
    nrows: int = 1
    ncols: int = 1

    def __init__(self, source: pd.DataFrame, target: str, **kwds) -> None:
        self.source = source
        self.target = target
        self.axes_facets = AxesFacets(self.nrows, self.ncols, **kwds)
        self._data: pd.DataFrame | None = None

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements"""
        return self.axes_facets.figure
    
    @property
    def axes(self) -> np.ndarray:
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

    def __init__(self, source: pd.DataFrame, target: str, feature: str = '',
            hue: str = '', dodge: bool = False, 
            categorical_features: bool = False, **kwds) -> None:
        self.categorical_features = categorical_features or dodge
        self.feature = feature
        self.hue = hue
        super().__init__(source=source, target=target, **kwds)
        self.coloring = HueLabel(self.get_categorical_labels(self.hue))
        feature_tick_labels = ()
        if self.categorical_features:
            assert feature in source
            feature_tick_labels = self.get_categorical_labels(feature)
        self.dodging = Dodger(self.coloring.labels,feature_tick_labels)
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
        _pos = PLOTTER.DEFAULT_POS
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
            self, plotter: _Plotter, target_on_y: bool = True, **kwds) -> Self:
        self.target_on_y = target_on_y
        for data in self:
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=target_on_y, color=self.color, 
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

    def __init__(
            self, source: pd.DataFrame, target: str, feature: str,
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
        self._sizes: np.ndarray | None = None
        self._variate_names = (self.hue, self.shape)
        self._reset_variate_()
    
    @property
    def marker(self) -> str:
        """Get marker for current variate"""
        marker_variate = self._current_variate.get(self.shape, None)
        return self.marking[marker_variate]

    @property
    def sizes(self) -> np.ndarray | None:
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
            self, plotter: _Plotter, target_on_y: bool = True, **kwds) -> Self:
        self.target_on_y = target_on_y
        for data in self:
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=target_on_y, color=self.color, 
                ax=self.axes_facets.ax, marker=self.marker, size=self.sizes,
                **kwds)
            plot()
        return self


class MultipleVariateChart(RelationalChart):

    target_on_y = True

    def __init__(
            self, source: pd.DataFrame, target: str, feature: str = '',
            hue: str = '', shape: str = '', size: str = '', col: str = '',
            row: str = '', dodge: bool = False
            ) -> None:
        self.source = source
        self.col = col
        self.row = row
        self.row_labels = self.get_categorical_labels(self.row)
        self.col_labels = self.get_categorical_labels(self.col)
        self.nrows = max([1, len(self.row_labels)])
        self.ncols = max([1, len(self.col_labels)])
        super().__init__(
            source=self.source, target=target, feature=feature, hue=hue, 
            shape=shape, size=size, dodge=dodge, sharex=True, sharey=True)
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
        self.plotter = plotter
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
    MultipleVariateChart.__name__
]