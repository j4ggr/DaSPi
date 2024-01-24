import numpy as np
import pandas as pd

from typing import Any
from typing import Self
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import Generator
from pathlib import Path

from .utils import HueLabelHandler
from .utils import SizeLabelHandler
from .utils import ShapeLabelHandler
from .facets import LabelFacets
from .facets import CategoricalAxesFacets
from .plotter import Scatter
from .plotter import BasePlotter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._constants import KW
from .._constants import COLOR


class MultipleVariateChart:

    marker_size_min: int = 2
    marker_size_max: int = 10
    n_marker_sizes: 4

    def __init__(
            self, source: pd.DataFrame, target: str, feature: str = '',
            hue: str = '', shape: str = '', size: str = '', col: str = '',
            row: str = ''
            ) -> None:
        self.source = source
        self.target = target
        self.feature = feature
        self.shape = shape
        self.hue = hue
        self.size = size
        self.col = col
        self.row = row
        self.axes_facets = CategoricalAxesFacets(
            source=self.source, col=self.col, row=self.row)
        
        self.colors = HueLabelHandler(self._labels_(self.hue))
        self.markers = ShapeLabelHandler(self._labels_(self.shape))
        self.sizes = None
        if self.size:
            self.sizes = SizeLabelHandler(
                self.source[self.size].min(), self.source[self.size].max())
        
        self._variate_names = (self.row, self.col, self.hue, self.shape)
        self._current_variate = {k: None for k in self.variate_names}
        self.next_ax: bool = False

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
        key = self._current_variate.get(self.hue, None)
        return self.colors[key]
    
    @property
    def marker(self) -> str:
        key = self._current_variate.get(self.shape, None)
        return self.markers[key]

    def _labels_(self, colname: str) -> Tuple:
        """Get sorted unique elements of given column in source"""
        return self.axes_facets._categorical_labels_(colname)
    
    def __iter__(self) -> Generator[Tuple, Self, None]:
        def variate_gen() -> Generator[Tuple, Self, None]:
            for combination, data in self.source.groupby(self.variate_names):
                if not isinstance(combination, tuple):
                    combination = (combination, )
                self.next_ax = self._row_or_col_changed_(combination)
                self.update_variate(combination)
                yield data
            self._current_variate = {k: None for k in self.variate_names}
            self.next_ax = False
        return variate_gen()
    
    def __next__(self) -> Axes:
        return next(self)

    def plot(self, plotter: BasePlotter):
        self._set_sizes_kind_(plotter)
        ax = None
        _ax = iter(self.axes_facets)
        for data in self:
            if self.next_ax or ax is None:
                ax = next(_ax)
            sizes = self.sizes(data[self.size]) if self.size else None
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                orientation='vertical', color=self.color, ax=ax, 
                marker=self.marker, sizes=sizes)
            plot()
                
    def label(
            self, fig_title: str = '', sub_title: str = '', xlabel: str = '',
            ylabel: str = '', row_title: str = '', col_title: str = '',
            info: bool | str = False):
        if self.row and not row_title: row_title = self.row
        if self.col and not col_title: col_title = self.col
        xlabel = xlabel if xlabel else self.feature
        ylabel = ylabel if ylabel else self.target
        
        handlers = (self.colors, self.markers, self.sizes)
        titles = (self.hue, self.shape, self.size)
        legends = {t: h.handles_labels() for t, h in zip(titles, handlers)}

        label = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            rows=self.axes_facets.row_labels, cols=self.axes_facets.col_labels,
            info=info, row_title=row_title, col_title=col_title,
            legends=legends)
        label.draw()

    def update_variate(self, combination: Tuple) -> None:
        for key, name in zip(self.variate_names, combination):
            self._current_variate[key] = name
    
    def _row_or_col_changed_(self, combination: Tuple) -> bool:
        """Check whether the current combination belongs to a new row 
        or column relative to the last combination"""
        for key, new in zip(self.variate_names, combination):
            old = self._current_variate[key]
            if old != new and key in (self.row, self.col):
                return True
        return False
    
    def _set_sizes_kind_(self, plotter: BasePlotter) -> None:
        """Set kind attribute of sizes according to given plotter"""
        if not self.sizes: return
        if isinstance(plotter, Scatter):
            self.sizes.kind = 'scatter'
        else:
            self.sizes.kind = 'line'
    
    def save(self, fname: str | Path, **kwds):
        kw = KW.SAVE_CHART | kwds
        self.figure.savefig(fname, **kw)
