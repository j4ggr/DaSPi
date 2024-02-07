import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Self
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import Generator
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.artist import Artist
from matplotlib.patches import Patch

from .._strings import STR
from .._constants import KW
from .._constants import LABEL


class LabelFacets:

    def __init__(
            self, figure: Figure, axes: NDArray, fig_title: str = '', 
            sub_title: str = '', xlabel: str | Tuple[str] = '',
            ylabel: str | Tuple[str] = '', info: bool | str = False,
            rows: Tuple[str] = (), cols: Tuple[str] = (),
            row_title: str = '', col_title: str = '',
            axes_titles: Tuple[str] = (), legends: Dict[str, List] = {}
            ) -> None:
        self.figure = figure
        self.plot_axes = axes
        self.fig_title = fig_title
        self.sub_title = sub_title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.rows = rows
        self.cols = cols
        self.row_title = row_title
        self.col_title = col_title
        self.info = info
        self.axes_titles = axes_titles
        self._legends = legends
        self._legend: Legend | None = None
    
    @property
    def shift_text_y(self) -> float:
        """Get offset to move text based on the fig height"""
        return LABEL.SHIFT_BASE / self.figure.get_figheight()
    
    @property
    def shift_text_x(self) -> float:
        """Get offset to move text based on the fig width"""
        return LABEL.SHIFT_BASE / self.figure.get_figwidth()
    
    @property
    def shift_fig_title(self) -> float:
        """Get offset in y direction for fig title"""
        labels = (self.axes_titles, self.col_title, self.sub_title)
        n = (LABEL.AXES_PADDING
             + sum(map(bool, labels)))
        return n * self.shift_text_y
    
    @property
    def shift_sub_title(self) -> float:
        """Get offset in y direction for sub title"""
        labels = (self.col_title, self.axes_titles)
        n = (LABEL.AXES_PADDING
             + LABEL.LABEL_PADDING * int(any(map(bool, labels)))
             + sum(map(bool, labels)))
        return n * self.shift_text_y
    
    @property
    def shift_legend(self) -> float:
        """Get offset in x direction for legend"""
        labels = (self.row_title, self.rows)
        n = (LABEL.AXES_PADDING 
             + LABEL.LABEL_PADDING * sum(map(bool, labels))
             + sum(map(bool, labels)))
        return n * self.shift_text_x

    @property
    def legend(self) -> Legend | None:
        """Get legend added to figure."""
        return self._legend

    @property
    def legend_box(self) -> Artist | None:
        """Get legend box holding title, handles and labels."""
        if not self.legend: return None
        return self.legend.get_children()[0]

    def add_legend(
            self, handles: List[Patch | Line2D], labels: List[str], title: str
            ) -> None:
        """Adds a legend at the right side of the figure. If there is 
        already one, the existing one is extended with the new one
        
        Parameters
        ----------
        title : str
            Legend title
        labels : list of str, optional
            The labels must be in the same order as the corresponding 
            plots were drawn. If no labels are given, the handles and 
            labels of the first axes are used.
        merkers : list of str, optional
            Use this argument to extend a existing hue legend with a 
            symbolic legend.
        """
        kw = KW.LEGEND
        bbox = kw['bbox_to_anchor']
        kw['bbox_to_anchor'] = (bbox[0] + self.shift_legend, bbox[1])
        legend = Legend(
            self.plot_axes[0, -1], handles, labels, title=title, **kw)
        if not self.legend:
            self.figure.legends.append(legend)
            self._legend = legend
        else:
            new_children = legend.get_children()[0].get_children()
            self.legend_box.get_children().extend(new_children)
    
    def add_xlabel(self) -> None:
        if not self.xlabel: return
        if isinstance(self.xlabel, str):
            kw = KW.XLABEL
            kw['y'] = kw['y'] - LABEL.AXES_PADDING * self.shift_text_y
            self.figure.text(s=self.xlabel, **kw)
        else:
            for ax, xlabel in zip(self.plot_axes.flat, self.xlabel):
                if (len(ax.xaxis._get_shared_axis()) == 1 
                    or ax in self.plot_axes[-1]): 
                    ax.set(xlabel=xlabel)

    def add_ylabel(self) -> None:
        if not self.ylabel: return
        if isinstance(self.xlabel, str):
            kw = KW.YLABEL
            kw['x'] = kw['x'] - LABEL.AXES_PADDING * self.shift_text_x
            self.figure.text(s=self.ylabel, **kw)
        else:
            for ax, ylabel in zip(self.plot_axes.flat, self.ylabel):
                if (len(ax.yaxis._get_shared_axis()) == 1 
                    or ax in self.plot_axes.T[-1]): 
                    ax.set(ylabel=ylabel)

    def add_row_labels(self) -> None:
        """Add row labels and row title"""
        if not self.rows: return
        for axs, label in zip(self.plot_axes, self.rows):
            ax = axs[-1]
            kwds = KW.ROW_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.row_title, **KW.ROW_TITLE)
    
    def add_col_labels(self) -> None:
        """Add column labels and column title"""
        if not self.cols: return
        for ax, label in zip(self.plot_axes[0], self.cols):
            kwds = KW.COL_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.col_title, **KW.COL_TITLE)
    
    def add_axes_titles(self) -> None:
        """Add given titles to each axes."""
        if not self.axes_titles: return
        for ax, title in zip(self.plot_axes.flat, self.axes_titles):
            ax.set(title=title)

    def add_titles(self) -> None:
        """Add figure and sub title at the top."""
        if not self.fig_title and not self.sub_title: return

        kw_fig = KW.FIG_TITLE
        kw_sub = KW.SUB_TITLE
        kw_sub['y'] = kw_sub['y'] + self.shift_sub_title
        kw_fig['y'] = kw_fig['y'] + self.shift_fig_title
        if self.sub_title:
            self.figure.text(s=self.sub_title, **kw_sub)
        if self.fig_title:
            self.figure.text(s=self.fig_title, **kw_fig)
    
    def add_info(self) -> None:
        """Inserts an info text in the bottom left-hand corner of the 
        figure. By default, the info text contains today's date and the 
        user name. If self.info is a string, it is added to the 
        info text separated by a comma."""
        if not self.info: return
        info_text = f'{STR.TODAY} {STR.USERNAME}'
        if isinstance(self.info, str):
            info_text = f'{info_text}, {self.info}'
        kwds = KW.INFO
        if self.xlabel: kwds['y'] = kwds['y'] - self.shift_text_y
        self.figure.text(s=info_text, **kwds)
    
    def draw(self) -> None:
        self.add_xlabel()
        self.add_ylabel()
        self.add_row_labels()
        self.add_col_labels()
        for title, (handles, labels) in self._legends.items():
            self.add_legend(handles, labels, title)
        self.add_info()
        self.add_titles()


class AxesFacets:

    def __init__(
            self, nrows: int = 1, ncols: int = 1, sharex: str = 'none', 
            sharey: str = 'none', stretch_figsize: bool = True, **kwds
            ) -> None:
        """
        Parameters
        ----------
        nrows, ncols : int, default: 1
            Number of rows/columns of the subplot grid.

        sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
            Controls sharing of properties among x (*sharex*) or y 
            (*sharey*) axes:

            - True or 'all': x- or y-axis will be shared among all 
            subplots.
            - False or 'none': each subplot x- or y-axis will be 
            independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.

            When subplots have a shared x-axis along a column, only the 
            x tick labels of the bottom subplot are created. Similarly,
            when subplots have a shared y-axis along a row, only the 
            y tick labels of the first column subplot are created. To 
            later turn other subplots' ticklabels on, use 
            `~matplotlib.axes.Axes.tick_params`.

            When subplots have a shared axis that has units, calling
            `~matplotlib.axis.Axis.set_units` will update each axis with
            the new units.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets
            a relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width. 
            Equivalent to ``gridspec_kw={'width_ratios': [...]}``.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
            Convenience for ``gridspec_kw={'height_ratios': [...]}``.

        stretch_figsize : bool, optional
            If true, stretch the figure height and width based on the 
            number of rows and columns, by default True
        """
        figsize = kwds.pop('figsize', plt.rcParams['figure.figsize'])
        if stretch_figsize:
            figsize = ((1 + math.log(ncols, math.e)) * figsize[0],
                       (1 + math.log(nrows, math.e)) * figsize[1])
        self.figsize = figsize
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, 
            squeeze=False, figsize=self.figsize, **kwds)
        self.figure: Figure = fig
        self.axes: NDArray = axes
        self._ax: Axes | None = None
        self._nrows: int = nrows
        self._ncols: int = ncols
        if self.nrows == self.ncols == 1: self._ax = self.axes[0, 0]

    @property
    def ax(self) -> Axes:
        """Get the axes that is currently being worked on"""
        return self._ax
    
    @property
    def row_idx(self) -> int | None:
        """Get the index of the row from which the current axes 
        originates."""
        if self.ax is None: return None
        for i, axs in enumerate(self.axes):
            if self.ax in axs:
                return i
    
    @property
    def col_idx(self) -> int | None:
        """Get the index of the column from which the current axes 
        originates."""
        if self.ax is None: return None
        for i, axs in enumerate(self.axes.T):
            if self.ax in axs:
                return i
    
    @property
    def nrows(self) -> int:
        return self._nrows
    
    @property
    def ncols(self) -> int:
        return self._ncols
    
    def __iter__(self) -> Generator[Axes, Self, None]:
        def ax_gen() -> Generator[Axes, None, None]:
            for ax in self.axes.flat:
                self._ax = ax
                yield ax
            self._ax = None
        return ax_gen()
    
    def __next__(self) -> Axes:
        return next(self)
    
    def __getitem__(self, index: int):
        return self.axes.flat[index]


__all__ = [
    LabelFacets.__name__,
    AxesFacets.__name__,
]