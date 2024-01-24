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
from matplotlib.patches import Patch

from .._strings import STR
from .._constants import KW
from .._constants import COLOR


class LabelFacets:

    def __init__(
            self, figure: Figure, axes: np.ndarray, fig_title: str = '', 
            sub_title: str = '', xlabel: str | Tuple[str] = '',
            ylabel: str | Tuple[str] = '', info: bool | str = False,
            rows: Tuple[str] = (), cols: Tuple[str] = (),
            row_title: str = '', col_title: str = '',
            legends: Dict[str, List] = {}
            ) -> None:
        self.figure = figure
        self.axes = axes
        self.fig_title = fig_title
        self.sub_title = sub_title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.rows = rows
        self.cols = cols
        self.row_title = row_title
        self.col_title = col_title
        self._legends = legends
        self._info = info
        self._legend: Legend | None = None
    
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
        x_spines_left = self.axes[0][0].spines.left.get_window_extent().x1
        figure_width = self.figure.get_window_extent().width
        x_pos = x_spines_left/figure_width
        return dict(x=x_pos, ha='left')

    def _add_legend_(
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
        legend = Legend(
            self.axes[0][-1], handles, labels, title=title, **KW.LEGEND)
        if not self.legend:
            self.figure.legends.append(legend)
            self._legend = legend
        else:
            new_children = legend.get_children()[0].get_children()
            self.legend_box.get_children().extend(new_children)
    
    def _add_xlabel_(self) -> None:
        if not self.xlabel: return
        if len(self.axes[-1]) == 1:
            self.axes[-1][0].set_xlabel(self.xlabel)
        else:
            self.figure.text(s=self.xlabel, **KW.XLABEL)

    def _add_ylabel_(self) -> None:
        if not self.ylabel: return
        if len(self.axes) == 1:
            self.axes[0][0].set_ylabel(self.ylabel)
        else:
            self.figure.text(s=self.ylabel, **KW.YLABEL)

    def _add_row_labels_(self) -> None:
        """Add row labels and row title"""
        if not self.rows: return
        for axs, label in zip(self.axes, self.rows):
            ax = axs[-1]
            kwds = KW.ROW_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.row_title, **KW.ROW_TITLE)
    
    def _add_col_labels_(self) -> None:
        """Add column labels and column title"""
        if not self.cols: return
        for ax, label in zip(self.axes[0], self.cols):
            kwds = KW.COL_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.col_title, **KW.COL_TITLE)

    def _add_fig_title_(self) -> None:
        """Add figure title at top, aligned with the left side of first 
        axes. This change should be the very last for the whole figure, 
        otherwise the alignment is not guaranteed. When this method
        is called, the figure will be rendered."""
        if not self.fig_title: return
        self.figure.suptitle(self.fig_title, **self._title_left_params_())

    def _add_sub_title_(self) -> None:
        """Add sub title as axes title of top left axes object"""
        if not self.sub_title: return
        self.axes[0][0].set_title(self.sub_title, **KW.SUB_TITLE)
    
    def _add_info_(self) -> None:
        """Inserts an info text in the bottom left-hand corner of the 
        figure. By default, the info text contains today's date and the 
        user name. If self.info is a string, it is added to the 
        info text separated by a comma."""
        if not self.info: return
        info_text = f'{STR.TODAY} {STR.USERNAME}'
        if isinstance(self.info, str):
            info_text = f'{info_text}, {self.info}'
        self.figure.text(s=info_text, **KW.INFO)
    
    def draw(self):
        self._add_xlabel_()
        self._add_ylabel_()
        self._add_row_labels_()
        self._add_col_labels_()
        for title, (handles, labels) in self._legends.items():
            self._add_legend_(handles, labels, title)
        self._add_sub_title_()
        self._add_fig_title_()


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
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, 
            squeeze=False, **kwds)
        self.figure: Figure = fig
        self.axes: np.ndarray = axes
        self._ax: Axes | None = None
        self._nrows: int = nrows
        self._ncols: int = ncols

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
    

class CategoricalAxesFacets(AxesFacets):

    def __init__(
            self, source: pd.DataFrame, col: str = '', row: str = '') -> None:
        self.source: pd.DataFrame = source
        self.row: str = row
        self.col: str = col
        self.col_labels: Tuple = self._categorical_labels_(self.col)
        self.row_labels: Tuple = self._categorical_labels_(self.row)

        super().__init__(
            nrows=self.nrows, ncols=self.ncols, 
            sharex='all', sharey='all')

    @property
    def nrows(self) -> int:
        return max([1, len(self.row_labels)])

    @property
    def ncols(self) -> int:
        return max([1, len(self.col_labels)])

    def _categorical_labels_(self, colname: str) -> Tuple:
        """Get sorted unique elements of given column in source"""
        if not colname: return ()
        return tuple(sorted(np.unique(self.source[colname])))

__all__ = [
    LabelFacets.__name__,
    AxesFacets.__name__,
    CategoricalAxesFacets.__name__,
]