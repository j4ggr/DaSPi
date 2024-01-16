import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Self
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import Generator
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.artist import Artist

from .._strings import STR
from .._constants import KW
from .._constants import COLOR


class LabelFacets:

    def __init__(
            self, figure: Figure, fig_title: str = '', sub_title: str = '', 
            info: bool = False) -> None:
        self.figure: Figure = figure
        self.fig_title: str = fig_title
        self.sub_title: str = sub_title
        self._legend: Legend | None = None
        self._info: bool = info

    @property
    def axes(self) -> List[Axes]:
        """List of Axes in the Figure. You can access and modify the 
        Axes in the Figure through this list."""
        return self.figure.axes
    
    @property
    def legend(self) -> Legend | None:
        """Get legend added to figure"""
        return self._legend

    @property
    def legend_box(self) -> Artist | None:
        """Get legend box holding title, handles and labels"""
        if not self.legend: return None
        return self.legend.get_children()[0]
    
    def _title_left_params_(self) -> Dict:
        """Get params to set figure titel on the left side. The title 
        is then adjusted to the first axes left spine. When this method
        is called, the figure will be rendered.
        
        Parameters
        ----------
        ax : matplotlib Axes
            Axes object in the top left corner of figure
        
        Returns
        -------
        dict: 
        - 'x': float (as x coordinate for title)
        - 'ha': 'left'
        """
        self.figure.canvas.draw()
        x_spines_left = self.axes[0].spines.left.get_window_extent().x1
        figure_width = self.figure.get_window_extent().width
        x_pos = x_spines_left/figure_width
        return dict(x=x_pos, ha='left')

    def add_legend(
            self, title: str, labels: List[str] = [], markers: List[str] = []
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
        if not labels:
            handles, labels = self.axes[0].get_legend_handles_labels()
        else:
            n_labels = len(labels)
            colors = ['k'] * n_labels if markers else COLOR.PALETTE[:n_labels]
            if not markers: 
                markers = ['o' for _ in range(len(labels))]
            handles = [self._handle_(m, c) for m, c in zip(markers, colors)]
        legend = Legend(
            self.figure, handles, labels, title=title, **KW.LEGEND)
        if not self.legend:
            self.figure.legends.append(legend)
            self._legend = legend
        else:
            self.legend_box.extend(legend.get_children()[0].get_children())
    
    def add_info(self, additional_info: str = '') -> None:
        """Inserts an info text in the bottom left-hand corner of the 
        figure. By default, the info text contains today's date and the 
        user name. If "additional_info" is given, it is added to the 
        info text separated by a comma."""
        info_text = f'{STR.TODAY} {STR.USERNAME}'
        if additional_info:
            info_text = f'{info_text}, {additional_info}'
        self.figure.text(s=info_text, **KW.INFO)

    def add_fig_title(self, fig_title: str = '') -> None:
        """Add figure at top, aligned with the left side of first axes.
        This change should be the very last for the whole figure, 
        otherwise the alignment is not guaranteed. When this method
        is called, the figure will be rendered."""
        if not fig_title: 
            fig_title = self.fig_title
        self.figure.suptitle(fig_title, **self._title_left_params_())
        
    @classmethod
    def _handle_(marker: str, color: str | List) -> Line2D:
        'Create an Artist (Line2D with no data) for the legend'
        return Line2D([], [], linewidth=0, marker=marker, color=color)


class AxesFacets:

    def __init__(
            self, nrows: int = 1, ncols: int = 1, sharex: str = 'none', 
            sharey: str = 'none', **kwds
            ) -> None:
        """
        Parameters
        ----------
        nrows, ncols : int, default: 1
            Number of rows/columns of the subplot grid.

        sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
            Controls sharing of properties among x (*sharex*) or y (*sharey*)
            axes:

            - True or 'all': x- or y-axis will be shared among all subplots.
            - False or 'none': each subplot x- or y-axis will be independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.

            When subplots have a shared x-axis along a column, only the x tick
            labels of the bottom subplot are created. Similarly, when subplots
            have a shared y-axis along a row, only the y tick labels of the first
            column subplot are created. To later turn other subplots' ticklabels
            on, use `~matplotlib.axes.Axes.tick_params`.

            When subplots have a shared axis that has units, calling
            `~matplotlib.axis.Axis.set_units` will update each axis with the
            new units.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.  Equivalent
            to ``gridspec_kw={'width_ratios': [...]}``.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height. Convenience
            for ``gridspec_kw={'height_ratios': [...]}``.
        """
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, 
            squeeze=False, **kwds)
        self.figure: Figure = fig
        self.axs: np.ndarray = axs
        self._ax: Axes | None = None
        self._nrows: int = nrows
        self._ncols: int = ncols

    @property
    def ax(self) -> Axes:
        """Get the axes that is currently being worked on"""
        return self._ax
    
    @property
    def row_idx(self) -> int:
        """Get the index of the row from which the current axes 
        originates."""
        for i, axs in enumerate(self.axs):
            if self.ax in axs:
                return i
    
    @property
    def col_idx(self) -> int:
        """Get the index of the column from which the current axes 
        originates."""
        for i, axs in enumerate(self.axs.T):
            if self.ax in axs:
                return i
    
    @property
    def nrows(self) -> int:
        return self._nrows
    
    @property
    def ncols(self) -> int:
        return self._ncols
    
    def __iter__(self) -> Generator[Axes, Self, None]:
        def ax_gen() -> Generator[Axes, Self, None]:
            for ax in self.axs.flat:
                self._ax = ax
                yield ax
            self._ax = None
        return ax_gen()
    
    def __next__(self) -> Axes:
        return next(self)
    

class CategoricalAxesFacets(AxesFacets):

    def __init__(
            self, source: pd.DataFrame, col: str = '', row: str = '') -> None:
        self.source: pd.DataFrame = source
        self.row: str = row
        self.col: str = col
        self.col_labels: Tuple | None = self._categorical_labels_(self.col)
        self.row_labels: Tuple | None = self._categorical_labels_(self.row)

        super().__init__(
            nrows=len(self.row_names), ncols=len(self.col_names), sharex='all',
            sharey='all')
        
    @property
    def n_categories(self) -> Literal[4, 3, 2]:
        n: int = 2
        if self.col: n += 1
        if self.row: n += 1
        return n

    @property
    def row_title(self) -> str:
        return self.row_labels[self.row_idx] if self.row_labels else ''
        
    @property
    def col_title(self) -> str:
        return self.col_labels[self.col_idx] if self.col_labels else ''
    
    def label_current_row(self) -> None:
        """Label the row according to which category it corresponds to.
        Skip if current Axes is not one of the last column."""
        if not self.row_title | self.ax not in self.axes.T[-1]: return
        self.ax.text(0.5, 1, self.row_title)

    def label_current_column(self) -> None:
        """Label the column according to which category it corresponds 
        to. Skip if current Axes is not one of the first row."""
        if not self.col_title | self.ax not in self.axes[0]: return
        self.ax.text(1, 0, self.col_title)

    def _categorical_labels_(self, colname: str) -> Tuple | None:
        """Get sorted unique elements of given column in source"""
        if not colname: return
        return tuple(sorted(np.unique(self.source[colname])))

__all__ = [
    LabelFacets.__name__,
    AxesFacets.__name__,
    CategoricalAxesFacets.__name__,
]