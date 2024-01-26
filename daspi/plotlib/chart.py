import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Any
from typing import Self
from typing import Dict
from typing import List
from typing import Tuple
from typing import Generator
from pathlib import Path

from .utils import HueLabelHandler
from .utils import SizeLabelHandler
from .utils import ShapeLabelHandler
from .facets import LabelFacets
from .facets import AxesFacets
from .plotter import BasePlotter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._constants import KW


class XYChart:

    nrows: int = 1
    ncols: int = 1

    def __init__(
            self, source: pd.DataFrame, target: str, feature: str = '',
            hue: str = '', shape: str = '', size: str = '', **kwds):
        self.source = source
        self.target = target
        self.feature = feature
        self.shape = shape
        self.hue = hue
        self.size = size
        self._plotter: BasePlotter | None
        self.target_axis = 'y'
        
        self.coloring = HueLabelHandler(self.get_categorical_labels(self.hue))
        self.marking = ShapeLabelHandler(self.get_categorical_labels(self.shape))
        if self.size:
            self.sizing = SizeLabelHandler(
                self.source[self.size].min(), self.source[self.size].max())
        else:
            self.sizing = None
        self._sizes: np.ndarray | None = None
        
        self.axes_facets = AxesFacets(self.nrows, self.ncols, **kwds)

        self._variate_names = (self.hue, self.shape)
        self._current_variate = {}
        self._last_variate = {}
        self._reset_variate_()

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements"""
        return self.axes_facets.figure
    
    @property
    def axes(self) -> np.ndarray:
        """Get the created axes"""
        return self.axes_facets.axes
    
    @property
    def variate_names(self) -> List[str]:
        return [v for v in self._variate_names if v]
    
    @property
    def color(self) -> str:
        """Get color for current variate"""
        key = self._current_variate.get(self.hue, None)
        return self.coloring[key]
    
    @property
    def marker(self) -> str:
        """Get marker for current variate"""
        key = self._current_variate.get(self.shape, None)
        return self.marking[key]

    @property
    def sizes(self) -> np.ndarray | None:
        """Get sizes for current variate, is set in grouped data 
        generator."""
        return self._sizes
    
    @property
    def plotter(self) -> BasePlotter | None:
        """Get current plotter
        
        If setting plotter and got size, sizes.kind is also set"""
        return self._plotter
    @plotter.setter
    def plotter(self, plotter: BasePlotter):
        self._plotter = plotter
        if self.size: self.sizing.kind == self._plotter.kind
    
    @property
    def legend_handles_labels(self) -> Dict[str, Tuple[tuple]]:
        """Get dictionary of handles and labels
        - keys: titles as str
        - values: handles and labels as tuple of tuples"""
        handlers = (self.coloring, self.marking, self.sizing)
        titles = (self.hue, self.shape, self.size)
        return {t: h.handles_labels() for t, h in zip(titles, handlers) if t}

    def get_categorical_labels(self, colname: str) -> Tuple:
        """Get sorted unique elements of given column in source"""
        if not colname: return ()
        return tuple(sorted(np.unique(self.source[colname])))
    
    def get_xy_labels(
            self, xlabel: bool | str = '', ylabel: bool | str = ''
            ) -> Tuple[str]:
        """Get x and y axis labels for labels facets.
        
        Parameters
        ----------
        xlabel, ylabel: bool or str
            If a string is passed, it will be taken. If True labels of 
            given feature and target name are used taking into account 
            the se target_axis attribute. If False, empty string is 
            used, by default ''
        
        Returns
        -------
        xlabel, ylabel: str
            labels for x and y axis
        """
        if isinstance(xlabel, bool):
            _xlabel = self.feature if self.target_axis == 'y' else self.target
            xlabel = _xlabel if xlabel == True else ''
        if isinstance(ylabel, bool):
            _ylabel = self.feature if self.target_axis == 'x' else self.target
            ylabel = _ylabel if ylabel == True else ''
        return xlabel, ylabel
    
    def _set_sizes_kind_(self, plotter: BasePlotter) -> None:
        """Set kind attribute of sizes according to given plotter"""
        if not self.size: return
        self.sizing.kind == plotter.kind
    
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

    def _grouped_data_gen_(self) -> Generator[Tuple, Self, None]:
        if self.variate_names:
            for combination, data in self.source.groupby(self.variate_names):
                self.update_variate(combination)
                if self.size:
                    self._sizes = self.sizing(data[self.size])
                yield data
        else:
            if self.size:
                self._sizes = self.sizing(self.source[self.size])
            yield self.source
        self._reset_variate_()
    
    def __iter__(self) -> Generator[Tuple, Self, None]:
        return self._grouped_data_gen_()
        
    def __next__(self) -> Axes:
        return next(self)

    def plot(self, plotter: BasePlotter, target_axis='y', **kwds) -> Self:
        self.plotter = plotter
        self.target_axis = target_axis
        for data in self:
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_axis='y', color=self.color, ax=self.axes_facets.ax, 
                marker=self.marker, size=self.sizes)
            plot()
        return self

    def label(
        self, fig_title: str = '', sub_title: str = '', xlabel: bool | str = '',
        ylabel: bool | str = '', info: bool | str = False) -> Self:
        xlabel, ylabel = self.get_xy_labels(xlabel, ylabel)

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            info=info, legends=self.legend_handles_labels)
        label.draw()
        return self
    
    def save(self, file_name: str | Path, **kwds) -> Self:
        kw = KW.SAVE_CHART | kwds
        self.figure.savefig(file_name, **kw)
        return self

    def close(self):
        """"Close figure"""
        plt.close(self.figure)


class MultipleVariateChart(XYChart):

    def __init__(
            self, source: pd.DataFrame, target: str, feature: str = '',
            hue: str = '', shape: str = '', size: str = '', col: str = '',
            row: str = ''
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
            shape=shape, size=size, sharex=True, sharey=True)
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

    def plot(self, plotter: BasePlotter) -> Self:
        self.plotter = plotter
        ax = None
        _ax = iter(self.axes_facets)
        for data in self:
            if self.row_or_col_changed or ax is None: ax = next(_ax)
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_axis='y', color=self.color, ax=ax, 
                marker=self.marker, size=self.sizes)
            plot()
        return self
                
    def label(
            self, fig_title: str = '', sub_title: str = '', xlabel: str = '',
            ylabel: str = '', row_title: str = '', col_title: str = '',
            info: bool | str = False) -> Self:
        xlabel, ylabel = self.get_xy_labels(xlabel, ylabel)

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            rows=self.row_labels, cols=self.col_labels,
            info=info, row_title=row_title, col_title=col_title,
            legends=self.legend_handles_labels)
        label.draw()
        return self

__all__ = [
    XYChart.__name__,
    MultipleVariateChart.__name__
]