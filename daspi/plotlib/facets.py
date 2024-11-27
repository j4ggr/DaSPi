from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any
from typing import Self
from typing import List
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Literal
from typing import overload
from typing import Sequence
from typing import Generator
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.artist import Artist
from matplotlib.typing import HashableList
from matplotlib.patches import Patch

from .plotter import StripeLine
from .plotter import StripeSpan

from .._typing import SpecLimit
from .._typing import NumericSample1D
from .._typing import ShareAxisProperty
from .._typing import LegendHandlesLabels

from ..strings import STR

from ..constants import KW
from ..constants import LABEL
from ..constants import DEFAULT

from ..statistics.estimation import ProcessEstimator


def flat_unique(nested: NDArray | List[List]) -> List:
    """Flatten the given array and return unique elements while
    preserving the order."""
    if isinstance(nested, list):
        nested = np.array(nested)
    return list(pd.Series(nested.flatten()).unique())


class LabelFacets: # TODO: add docstring description how flat, axes and iteration works then give som examples
    """
    A class for adding labels and titles to facets of a figure.

    Parameters
    ----------
    figure : Figure
        The figure to label.
    axes : 2DArray
        A 2D array of Axes instances. Setting squeeze=False when using 
        the `plt.subplots` method ensures that it is always a 2D array.
    fig_title : str, optional
        Main title that should be displayed at the top of the chart,
        by default ''.
    sub_title : str, optional
        Subtitle, which should appear directly below the main title and
        slightly smaller than it, by default ''.
    xlabel, ylabel: str or Tuple[str, ...], optional
        The axis label(s) of the figure. To label multiple axes with 
        different names, provide a tuple; otherwise, provide a string,
        by default ''.
    info : bool or str, optional
        Indicates whether to include an info text at the lower left 
        corner in the figure. The date and user are automatically added,
        by default False.
    rows: Tuple[str, ...], optional
        The row labels of the figure, by default ().
    cols: Tuple[str, ...], optional
        The column labels of the figure, by default ().
    row_title : str, optional
        The title of the rows, by default ''.
    col_title : str, optional
        The title of the columns, by default ''.
    axes_titles : Tuple[str, ...]
        Title for each Axes, usefull for JointCharts, by default ()
    legend_data : Dict[str, LegendHandlesLabels], optional
        The legends to be added to the figure. The key is used as the 
        legend title, and the values must be a tuple of tuples, where
        the inner tuple contains a handle as a Patch or Line2D artist
        and a label as a string, by default {}.
    """

    figure: Figure
    """The figure instance to label."""
    axes: NDArray
    """A 2D array containing the Axes instances of the figure or an 
    existing AxesFacets instance."""
    fig_title: str
    """The title to display at the top of the chart."""
    sub_title: str
    """The subtitle to display directly below the title of the chart."""
    xlabel: str | Tuple[str, ...]
    """The x-axis label(s) of the figure."""
    ylabel: str | Tuple[str, ...]
    """The y-axis label(s) of the figure."""
    info: bool | str
    """Indicates whether to include an info text in the figure."""
    rows: Tuple[str, ...]
    """The row labels of the figure."""
    cols: Tuple[str, ...]
    """The column labels of the figure."""
    row_title: str
    """The title of the rows."""
    col_title: str
    """The title of the columns."""
    axes_titles: Tuple[str, ...]
    """The titles of each axes."""
    legend_data: Dict[str, LegendHandlesLabels]
    """The legend_data to be added to the figure."""
    _legend: Legend | None
    """Figure legend if one is added"""

    def __init__(
            self,
            figure: Figure,
            axes: NDArray,
            fig_title: str = '', 
            sub_title: str = '',
            xlabel: str | Tuple[str, ...] = '',
            ylabel: str | Tuple[str, ...] = '',
            info: bool | str = False,
            rows: Tuple[str, ...] = (),
            cols: Tuple[str, ...] = (),
            row_title: str = '',
            col_title: str = '',
            axes_titles: Tuple[str, ...] = (),
            legend_data: Dict[str, LegendHandlesLabels] = {}
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
        self.axes_titles = axes_titles
        self.info = info
        self.legend_data = legend_data
        self._legend = None
    
    @staticmethod
    def get_legend_artists(legend: Legend) -> List[Artist]:
        """Get the inner children of a legend.

        Parameters
        ----------
        legend : Legend
            The legend object.

        Returns
        -------
        List[Artist]:
            The artists representing the inner children of the legend.
        """
        return legend.get_children()[0].get_children()

    @property
    def shift_text_y(self) -> float:
        """Get offset to move text based on the fig height (read-only)."""
        return LABEL.SHIFT_BASE / self.figure.get_figheight()

    @property
    def shift_text_x(self) -> float:
        """Get offset to move text based on the fig width (read-only)."""
        return LABEL.SHIFT_BASE / self.figure.get_figwidth()

    @property
    def shift_fig_title(self) -> float:
        """Get offset in y direction for fig title (read-only)."""
        labels = (self.axes_titles, self.col_title, self.sub_title)
        n = (LABEL.AXES_PADDING
             + sum(map(bool, labels)))
        return n * self.shift_text_y

    @property
    def shift_sub_title(self) -> float:
        """Get offset in y direction for sub title (read-only)."""
        labels = (self.col_title, self.axes_titles)
        n = (LABEL.AXES_PADDING
             + LABEL.LABEL_PADDING * int(any(map(bool, labels)))
             + sum(map(bool, labels)))
        return n * self.shift_text_y

    @property
    def shift_legend(self) -> float:
        """Get offset in x direction for legend (read-only)."""
        labels = (self.row_title, self.rows)
        n = (LABEL.AXES_PADDING 
             + LABEL.LABEL_PADDING * sum(map(bool, labels))
             + sum(map(bool, labels)))
        return n * self.shift_text_x

    @property
    def legend(self) -> Legend | None:
        """Get legend added to figure (read-only)."""
        return self._legend
    
    @property
    def legend_artists(self) -> List[Artist]:
        """Get legend artists (read-only)."""
        if self.legend is None:
            return []
        return self.get_legend_artists(self.legend)

    def add_legend(
            self, handles: Tuple[Patch |Line2D, ...], labels: Tuple[str, ...],
            title: str) -> None:
        """Adds a legend at the right side of the figure. If there is 
        already one, the existing one is extended with the new one
        
        Parameters
        ----------
        handles: Tuple[Patch | Line2D, ...]
            A list of Artists (lines, patches) to be added to the
            legend.
        labels : Tuple[str, ...]
            The labels must be in the same order as the corresponding 
            plots were drawn. If no labels are given, the handles and 
            labels of the first axes are used.
        title : str
            Title for the given handles and labels. 
        """
        kw = KW.LEGEND
        bbox = kw['bbox_to_anchor']
        kw['bbox_to_anchor'] = (bbox[0] + self.shift_legend, bbox[1])
        legend = Legend(
            self.axes[0, -1], handles, labels, title=title, **kw)
        if not self.legend:
            self.figure.legends.append(legend) # type: ignore
            self._legend = legend
        else:
            new_artists = self.get_legend_artists(legend)
            self.legend_artists.extend(new_artists)

    def add_xlabel(self) -> None:
        """Add x-axis label(s) to the figure."""
        if not self.xlabel:
            return
        
        if isinstance(self.xlabel, str):
            kw = KW.XLABEL
            kw['y'] = kw['y'] - LABEL.AXES_PADDING * self.shift_text_y
            self.figure.text(s=self.xlabel, **kw)
        else:
            for ax, xlabel in zip(self.axes.flat, self.xlabel):
                if (len(ax.xaxis._get_shared_axis()) == 1 # type: ignore
                        or ax in self.axes[-1, :]): 
                    ax.set(xlabel=xlabel)

    def add_ylabel(self) -> None:
        """Add y-axis label(s) to the figure."""
        if not self.ylabel:
            return
        
        if isinstance(self.ylabel, str):
            kw = KW.YLABEL
            kw['x'] = kw['x'] - LABEL.AXES_PADDING * self.shift_text_x
            self.figure.text(s=self.ylabel, **kw)
        else:
            for ax, ylabel in zip(self.axes.flat, self.ylabel):
                if (len(ax.yaxis._get_shared_axis()) == 1 # type: ignore
                        or ax in self.axes[:, 0]): 
                    ax.set(ylabel=ylabel)

    def add_row_labels(self) -> None:
        """Add row labels and row title to the figure."""
        if not self.rows:
            return
        
        for ax, label in zip(self.axes[:, -1], self.rows):
            kwds = KW.ROW_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.row_title, **KW.ROW_TITLE)
    
    def add_col_labels(self) -> None:
        """Add column labels and column title to the figure."""
        if not self.cols:
            return
        for ax, label in zip(self.axes[0, :], self.cols):
            kwds = KW.COL_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.col_title, **KW.COL_TITLE)

    def add_titles(self) -> None:
        """Add the figure and sub-title at the top of the chart."""
        if not self.fig_title and not self.sub_title:
            return
        kw_fig = KW.FIG_TITLE
        kw_sub = KW.SUB_TITLE
        kw_sub['y'] = kw_sub['y'] + self.shift_sub_title
        kw_fig['y'] = kw_fig['y'] + self.shift_fig_title
        if self.sub_title:
            self.figure.text(s=self.sub_title, **kw_sub)
        if self.fig_title:
            self.figure.text(s=self.fig_title, **kw_fig)
    
    def add_axes_titles(self) -> None:
        """Add the provided axes titles."""
        if not self.axes_titles:
            return
        
        for ax, title in zip(self.axes.flat, self.axes_titles):
            ax.set_title(title)
    
    def add_info(self) -> None:
        """Insert an info text in the bottom left-hand corner of the 
        figure. By default, the info text contains today's date and the 
        user name. If attribute `info` is a string, it is added to the 
        info text separated by a comma."""
        if not self.info:
            return
        
        info_text = f'{STR.TODAY} {STR.USERNAME}'
        if isinstance(self.info, str):
            info_text = f'{info_text}, {self.info}'
        kwds = KW.INFO
        if self.xlabel:
            kwds['y'] = kwds['y'] - self.shift_text_y
        self.figure.text(s=info_text, **kwds)
    
    def draw(self) -> None:
        """Draw all the label facets to the figure."""
        self.add_xlabel()
        self.add_ylabel()
        self.add_axes_titles()
        self.add_row_labels()
        self.add_col_labels()
        for title, (handles, labels) in self.legend_data.items():
            self.add_legend(handles, labels, title)
        self.add_info()
        self.add_titles()


class AxesFacets:
    """A class for creating a grid of subplots with customizable sharing
    and sizing options.

    Parameters
    ----------
    nrows : int, optional
        Number of rows of subplots in the grid, by default 1.
    ncols : int, optional
        Number of columns of subplots in the grid, by default 1.
    sharex : bool or {'none', 'all', 'row', 'col'}, optional
        Controls sharing of properties along the x-axis,
        by default 'none'.
    sharey : bool or {'none', 'all', 'row', 'col'}, optional
        Controls sharing of properties along the y-axis,
        by default 'none'.
    width_ratios : array-like of length ncols, optional
        Relative widths of the columns, by default None.
    height_ratios : array-like of length nrows, optional
        Relative heights of the rows, by default None.
    stretch_figsize : bool, optional
        If True, stretch the figure height and width based on the number of
        rows and columns, by default False.
    **kwds : dict, optional
        Additional keyword arguments to pass to the `plt.subplots` function.
    """

    figsize: Tuple[float, float]
    """The figsize passed when creating the subplots."""
    figure: Figure
    """The figure instance to label."""
    axes: NDArray
    """A 2D array containing the Axes instances of the figure."""
    _nrows: int
    """Number of Axes rows in the grid"""
    _ncols: int
    """Number of Axes columns in the grid"""
    _sharex: ShareAxisProperty
    """Controls sharing of properties along the x-axis."""
    _sharey: ShareAxisProperty
    """Controls sharing of properties along the y-axis."""
    _ax: Axes | None
    """The current axes being worked on"""
    _default_ax: Axes | None
    """The default Axes instance after iterating over this class. It is 
    the first Axes object in the grid if there is only one, otherwise 
    None."""
    mosaic: List[HashableList] | None
    """A visual layout of how the Axes are arranged labeled as strings."""

    def __init__(
            self,
            nrows: int | None = None,
            ncols: int | None = None,
            mosaic: List[HashableList] | str | None = None,
            sharex: ShareAxisProperty = 'none', 
            sharey: ShareAxisProperty = 'none', 
            width_ratios: Sequence[float] | None = None,
            height_ratios: Sequence[float] | None = None, 
            stretch_figsize: bool = False,
            **kwds
            ) -> None:
        assert not all(arg is not None for arg in (nrows, ncols, mosaic)), (
            'Either nrows and ncols or mosaic must be provided, but not both.')
        
        if isinstance(mosaic, str):
            mosaic = [list(r.strip()) for r in mosaic.strip('\n').split('\n')]
        self.mosaic = mosaic
        
        if self.mosaic is not None:
            self._nrows = len(self.mosaic)
            self._ncols = len(self.mosaic[0])
        else:
            self._nrows = nrows if isinstance(nrows, int) else 1
            self._ncols = ncols if isinstance(ncols, int) else 1

        figsize = kwds.pop('figsize', plt.rcParams['figure.figsize'])
        if stretch_figsize:
            figsize = (
                (1 + math.log(self._ncols, math.e)) * figsize[0],
                (1 + math.log(self._nrows, math.e)) * figsize[1])
        self.figsize = figsize

        if self.mosaic is not None:
            assert all(self._ncols == len(row) for row in self.mosaic), (
                'Mosaic must be a rectangular grid of strings or hashes.')
            self.figure, axes = plt.subplot_mosaic(
                mosaic=self.mosaic,
                sharex=True if sharex in (True, 'all') else False,
                sharey=True if sharey in (True, 'all') else False,
                height_ratios=width_ratios,
                width_ratios=width_ratios,
                **kwds)
            self.axes = np.array(
                [[axes[key] for key in row] for row in self.mosaic])
        else:
            self.figure, self.axes = plt.subplots(
                nrows=self._nrows,
                ncols=self._ncols,
                sharex=sharex,
                sharey=sharey, 
                squeeze=False,
                figsize=self.figsize,
                width_ratios=width_ratios,
                height_ratios=height_ratios, **kwds,)

        self._sharex = sharex
        self._sharey = sharey
        if self.nrows == self.ncols == 1:
            self._default_ax = self.axes[0, 0] 
        else:
            self._default_ax = None
        self._ax = self._default_ax

    @property
    def ax(self) -> Axes | None:
        """Get the axes that is currently being worked on. This property
        is automatically kept current when iterating through this 
        class (read-only)."""
        return self._ax
    
    @property
    def nrows(self) -> int:
        """Get the number of Axes rows in the grid (read-only)."""
        return self._nrows
    
    @property
    def ncols(self) -> int:
        """Get the number of Axes columns in the grid (read-only)."""
        return self._ncols
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the axes array (read-only)."""
        return self.axes.shape

    @property
    def sharex(self) -> ShareAxisProperty:
        """Get the sharing of properties along the x-axis (read-only)."""
        return self._sharex

    @property
    def sharey(self) -> ShareAxisProperty:
        """Get the sharing of properties along the y-axis (read-only)."""
        return self._sharey
    
    @property
    def flat(self) -> List[Axes]:
        return flat_unique(self.axes)
    
    def __iter__(self) -> Generator[Axes, Self, None]:
        """Iterate over the axes in the grid.

        Returns
        -------
        Generator[Axes, Self, None]
            The generator object that yields the axes in the grid.
        """
        def ax_gen(self) -> Generator[Axes, Self, None]:
            for ax in self.flat:
                self._ax = ax
                yield ax
            self._ax = self._default_ax
        return ax_gen(self)
    
    def __next__(self) -> Axes:
        """Get the next axes in the grid.

        Returns
        -------
        Axes
            The next axes in the grid."""
        return next(self)
    
    @overload
    def __getitem__(self, index: Tuple[slice, slice]) -> NDArray:...

    @overload
    def __getitem__(self, index: Tuple[int, slice]) -> NDArray:...

    @overload
    def __getitem__(self, index: Tuple[slice, int]) -> NDArray:...
    
    @overload
    def __getitem__(self, index: int | Tuple[int, int]) -> Axes:...
    
    def __getitem__(self, index: int | Tuple[int | slice, int | slice]) -> Any:
        """Get the axes at the specified index in the grid. If the index
        is a number, the axes are fetched from left to right and from 
        top to bottom. If two numbers are specified, the axis is fetched 
        directly in the grid and it is then also possible to fetch an 
        axis across multiple indices if the axis is spanned across 
        multiple columns or rows.

        Parameters
        ----------
        index : int | Tuple[int, int]
            The index of the axes in the grid.

        Returns
        -------
        Axes
            The axes at the given index in the grid.
        """
        if isinstance(index, tuple):
            return self.axes[index]
        return self.flat[index]

    def __len__(self) -> int:
        """Get the total number of axes in the grid.

        Returns
        -------
        int
            The total number of axes in the grid.
        """
        return len(self.flat)


class StripesFacets:
    """A class for creating location and spread width lines, 
    specification limits and/or confidence interval areas as stripes on 
    each Axes. The location and spread (and their confidence bands) 
    represent the data per axes.

    Parameters
    ----------
    target : ArrayLike
        The target data for the estimation.
    single_axes : bool
        Whether a single axis is used for the chart or whether the 
        stripes are used on multiple axes. This affects how the 
        predefined legend labels are handled. If True, the corresponding
        position values are added to the label.
    stripes : List[StripeLine | StripeSpan], optional
        Additional non-predefined stripes to be added to the chart, 
        by default [].
    mean : bool, optional
        Whether to include a mean line on the chart, by default False.
    median : bool, optional
        Whether to include a median line on the chart, by default False.
    control_limits : bool, optional
        Whether to include control limits on the chart, by default False.
    spec_limits : Tuple[float | None, float | None], optional
        The specification limits for the chart, by default (None, None).
    confidence : float, optional
        The confidence level for the confidence intervals,
        by default None.
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the control limits
        (process spread):
        - eval: The strategy is determined according to the given 
          evaluate function. If none is given, the internal evaluate
          method is used.
        - fit: First, the distribution that best represents the process
          data is searched for and then the agreed process spread is 
          calculated
        - norm: it is assumed that the data is subject to normal
          distribution. The variation tolerance is then calculated as 
          agreement * standard deviation
        - data: The quantiles for the process variation tolerance are
          read directly from the data. by default 'norm'
    
    agreement : float or int, optional
        Specify the tolerated process variation for which the control 
        limits are to be calculated.
        If int, the spread is determined using the normal distribution 
        agreementsigma, e.g. agreement = 6 -> 6sigma ~ covers 99.75 % of 
        the data. The upper and lower permissible quantiles are then 
        calculated from this.
        If float, the value must be between 0 and 1.This value is then 
        interpreted as the acceptable proportion for the spread, e.g. 
        0.9973 (which corresponds to ~ 6 sigma) by default 6
    **kwds : dict, optional
        Additional keyword arguments to pass to the ProcessEstimator.
    """

    __slots__ = (
        'estimation', '_confidence', 'single_axes', 'stripes', 'target_on_y')
    
    estimation: ProcessEstimator
    """The process estimator object."""
    _confidence: float | None
    """The confidence level for the confidence intervals."""
    single_axes: bool
    """Whether a single axis is used for the chart or whether the 
    stripes are used on multiple axes. This affects how the predefined
    legend labels are handled. If True, the corresponding position
    values are added to the label."""
    stripes: Dict[str, StripeLine | StripeSpan]
    """The stripes to plot on the chart."""
    target_on_y: bool
    """Whether the target is on the y-axis or the x-axis."""
    
    def __init__(
        self,
        target: NumericSample1D,
        target_on_y: bool,
        single_axes: bool,
        stripes: List[StripeLine | StripeSpan] = [], 
        mean: bool = False,
        median: bool = False,
        control_limits: bool = False,
        spec_limits: Tuple[SpecLimit, SpecLimit] = (None, None),
        confidence: float | None = None,
        strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
        agreement: float | int = 6,
        **kwds) -> None:
        assert len(spec_limits) == 2, (
            f'Specification limits must contain 2 values, got {spec_limits}. '
            'Set None for limits that do not exist')
        self.single_axes = single_axes
        self._confidence = confidence
        self.estimation = ProcessEstimator(
            samples=target,
            lsl=self.na_to_none(spec_limits[0]),
            usl=self.na_to_none(spec_limits[1]),
            strategy=strategy,
            agreement=agreement,
            **kwds)
        self.target_on_y = target_on_y
        self.stripes = {s.identity: s for s in stripes}
        add_confidence_span = confidence is not None
        
        self.add_specification_limit_stripes()
        if mean:
            self.add_mean_stripes(add_confidence_span)
        if median:
            self.add_median_stripes(add_confidence_span)
        if control_limits:
            self.add_control_limit_stripes(add_confidence_span)
        
    @property
    def confidence(self) -> float:
        """Get the confidence level for the confidence intervals
        (read-only)."""
        if self._confidence is None:
            return DEFAULT.CONFIDENCE_LEVEL
        return self._confidence

    @staticmethod
    def na_to_none(value: Any) -> Any:
        """Convert NaN to None."""
        return None if pd.isna(value) else value
    
    @property
    def orientation(self) -> Literal['horizontal', 'vertical']:
        """The orientation of the chart."""
        return 'horizontal' if self.target_on_y else 'vertical'
    
    @property
    def _kwds(self) -> Dict[str, Any]:
        kwds = dict(
            orientation=self.orientation,
            show_position=self.single_axes)
        return kwds

    @property
    def _confidence_label(self) -> str:
        """The lable used for confidence stripes."""
        if self._confidence is not None:
            label = f'{100*self._confidence:.0f} \\%-{STR["ci"]}'
        else:
            label = ''
        return label
    
    def add_mean_stripes(self, add_confidence_span: bool) -> None:
        """Add mean stripes to the plot.

        This method adds a stripe representing the mean of the target 
        data, using the `StripeLine` class. If `add_confidence_span` is 
        `True`, additional stripes representing the confidence interval 
        for the mean will also be added using the `StripeSpan` class.

        Parameters
        ----------
        add_confidence_span : bool
            Whether to add confidence stripes for the mean.
        """
        line = StripeLine(
            label=r'\bar x',
            position=self.estimation.mean,
            **(self._kwds | KW.MEAN_LINE))
        self.stripes[line.identity] = line

        if add_confidence_span:
            low, upp = self.estimation.mean_ci(self.confidence)
            span = StripeSpan(
                label=self._confidence_label,
                lower_position=low,
                upper_position=upp,
                **(self._kwds | KW.STRIPES_CONFIDENCE))
            self.stripes[f'{span.identity}_mean'] = span

    def add_median_stripes(self, add_confidence_span: bool) -> None:
        """Add median stripes to the plot.

        This method adds a stripe representing the median of the target 
        data, using the `StripeLine` class. If `add_confidence_span` is 
        `True`, additional stripes representing the confidence interval 
        for the median will also be added using the `StripeSpan` class.

        Parameters
        ----------
        add_confidence_span : bool
            Whether to add confidence stripes for the median.
        """
        line = StripeLine(
            label=r'\tilde x',
            position=self.estimation.median,
            **(self._kwds | KW.MEDIAN_LINE))
        self.stripes[line.identity] = line

        if add_confidence_span:
            low, upp = self.estimation.median_ci(self.confidence)
            span = StripeSpan(
                label=self._confidence_label,
                lower_position=low,
                upper_position=upp,
                **(self._kwds | KW.STRIPES_CONFIDENCE))
            self.stripes[f'{span.identity}_median'] = span
    
    def add_control_limit_stripes(self, add_confidence_span: bool) -> None:
        """Add control limit stripes to the plot.
        
        This method adds stripes representing the lower and upper 
        control limits for the target data.
        
        The stripes are added using the `StripeLine` class, with the 
        label and position set based on the control limits from the 
        `estimation` attribute. If `add_confidence_span` is `True`,
        additional stripes representing the confidence interval for the
        control limits will also be added using the `StripeSpan` class.
        
        Parameters
        ----------
        add_confidence_span : bool
            Whether to add confidence stripes for the control limits.
        """
                
        kw_line = self._kwds | KW.CONTROL_LINE
        line_low = StripeLine(
            label=r'x_{' + f'{self.estimation.q_low:.4f}' + '}',
            position=self.estimation.lcl,
            **kw_line)
        line_upp = StripeLine(
            label=r'x_{' + f'{self.estimation.q_upp:.4f}' + '}',
            position=self.estimation.ucl,
            **kw_line)
        self.stripes[line_low.identity] = line_low
        self.stripes[line_upp.identity] = line_upp
        
        if add_confidence_span:
            kw_span = self._kwds | KW.STRIPES_CONFIDENCE
            low, upp = self.estimation.stdev_ci(self.confidence)
            span_low = StripeSpan(
                label=self._confidence_label,
                position=self.estimation.lcl,
                width=upp - low,
                **kw_span)
            span_upp = StripeSpan(
                label=self._confidence_label,
                position=self.estimation.ucl,
                width=upp - low,
                **kw_span)
            self.stripes[f'{span_low.identity}_low'] = span_low
            self.stripes[f'{span_upp.identity}_upp'] = span_upp
    
    def add_specification_limit_stripes(self) -> None:
        """Add specification limit stripes to the plot.
        
        This method adds stripes representing the lower and upper 
        specification limits to the plot. If a specification limit is
        `None`, it will be skipped.
        
        The stripes are added using the `StripeLine` class, with the 
        label and position set based on the specification limits from
        the `estimation` attribute.
        """
        kwds = self._kwds | KW.SPECIFICATION_LINE
        _labels = (STR['lsl'], STR['usl'])
        for label, limit in zip(_labels, self.estimation.limits):
            if limit is None:
                continue
            
            line = StripeLine(label=str(label), position=limit, **kwds)
            self.stripes[line.identity] = line
    
    def handles_labels(self) -> LegendHandlesLabels:
        """Get the legend handles and labels for the plot.
        
        This method processes the `self.stripes` dictionary to extract
        the legend handles and labels for the plot. It separates the 
        handles and labels for line stripes and span stripes, and 
        returns them as a tuple of handles and a tuple of labels.
        
        Returns
        -------
        LegendHandlesLabels:
            A tuple containing the legend handles and labels.
        """
        handles_labels_lines = {}
        handles_labels_spans = {}
        processed = set()
        for stripe in self.stripes.values():
            if stripe.identity in processed:
                continue
            if isinstance(stripe, StripeLine):
                handles_labels_lines[stripe.label] = stripe.handle
            else:
                handles_labels_spans[stripe.label] = stripe.handle
            processed.add(stripe.identity)
        handles_labels = handles_labels_lines | handles_labels_spans
        return tuple(handles_labels.values()), tuple(handles_labels.keys())

    def draw(self, ax: Axes) -> None:
        """Draw the stripes on the specified Axes.

        Parameters
        ----------
        ax : Axes
            The Axes on which to draw the stripes.
        """
        for stripe in self.stripes.values():
            stripe(ax)


__all__ = [
    "flat_unique",
    "LabelFacets",
    "AxesFacets",
    "StripesFacets",
]