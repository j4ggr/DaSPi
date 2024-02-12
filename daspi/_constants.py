import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from typing import List
from dataclasses import dataclass
from scipy.stats._continuous_distns import _distn_names


class _Kw_:
    _x: float = 0.035 # X position for fig title, subtitle and info
    @property
    def LINE(self):
        """Base kwds for horizontal or vertical lines"""
        return dict(ls=0.8, style='--')
    @property
    def HUE_HANDLES(self) -> dict:
        """Patch keyword arguments for genereting handles on 
        HueLabel"""
        return dict(alpha=COLOR.FILL_ALPHA)
    @property
    def SHAPE_HANDLES(self) -> dict:
        """Line2D keyword arguments for genereting handles on 
        SizeLabel"""
        return dict(xdata=[], ydata=[], c=COLOR.HANDLES, lw=0)
    @property
    def SIZE_HANDLES(self) -> dict:
        """Line2D keyword arguments for genereting handles on 
        ShapeLabel"""
        return dict(
            xdata=[], ydata=[], c=COLOR.HANDLES, marker='o', lw=0, 
            alpha=COLOR.MARKER_ALPHA)
    @property
    def LEGEND(self) -> dict:
        """Figure legend at right side of figure"""
        return dict(
            loc='upper left', bbox_to_anchor=(1, 1), alignment='left')
    @property
    def SAVE_CHART(self) -> dict:
        """Key word arguments for matplotlib savefig"""
        return dict(bbox_inches='tight')
    @property
    def XLABEL(self) -> dict:
        """Keyword arguments for Figure.text method used to add a 
        centered xlabel"""
        return dict(x=0.5, y=0, ha='center', va='top')
    @property
    def YLABEL(self) -> dict:
        """Keyword arguments for Figure.text method used to add a 
        centered xlabel"""
        return dict(x=0, y=0.5, ha='right', va='center', rotation=90)
    @property
    def ROW_LABEL(self) -> dict:
        """Keyword Arguments for the Axes.text method used to add a 
        row label to each plot_axes as text on LabelFacets."""
        return dict(x=1, y=0.5, ha='left', va='center', rotation=-90)
    @property
    def ROW_TITLE(self) -> dict:
        """Keyword Arguments for the Axes.text method used to add a 
        row title to ax as text on LabelFacets."""
        return self.ROW_LABEL | {'x': 1}
    @property
    def COL_LABEL(self) -> dict:
        """Keyword Arguments for the Axes.text method used to add a 
        column label to each plot axis as text on LabelFacets."""
        return dict(x=0.5, y=1, ha='center', va='bottom')
    @property
    def COL_TITLE(self) -> dict:
        """Keyword Arguments for the Figure.text method used to add a 
        column title to each plot axis as text on LabelFacets."""
        return self.COL_LABEL | {'y': 1}
    @property
    def FIG_TITLE(self) -> dict:
        """Keyword arguments for Figure.text method used for adding
        figure title on LabelFacets."""
        return dict(x=self._x, y=1, ha='left', size='x-large', va='bottom')
    @property
    def SUB_TITLE(self) -> dict:
        """Keyword arguments for Figure.set_title method used for adding
        sub title at LabelFacets."""
        return self.FIG_TITLE | dict(size='large')
    @property
    def INFO(self) -> dict:
        """Adding info text at bottom left of figure."""
        return dict(x=self._x, y=0, ha='left', va='top', size='x-small')
    @property
    def ERROR_BAR(self) -> dict:
        """Base keyword arguments for error bars"""
        return dict(color='k', lw=0.5, fmt='none')
KW = _Kw_()


@dataclass(frozen=True)
class _Color_:
    GOOD: str = '#2ca02c'
    BAD: str = '#d62728'
    MEAN: str = '#101010'
    PERCENTIL: str = '#303030'
    HANDLES: str = '#202020'
    TRANSPARENT: str = '#ffffff00'
    WHITE_TRANSPARENT: str = '#ffffffaa'
    MARKER_ALPHA: float = 0.5
    FILL_ALPHA: float = 0.5
    @property
    def PALETTE(self) -> List[str]:
        """Get prop cycler color palette"""
        return plt.rcParams['axes.prop_cycle'].by_key()['color']
    @property
    def LIMITS(self) -> Tuple[str]:
        """Color for specification limits"""
        return (self.BAD, self.BAD)
    @property
    def STATISTIC_LINES(self) -> Tuple[str]:
        """Statistic lines color in order lower (Percentil Q_0.99865)
        upper (Percentil Q_0.00135) and mean"""
        return (self.PERCENTIL, self.PERCENTIL, self.MEAN)
COLOR = _Color_()


@dataclass(frozen=True)
class _Label_:
    SHIFT_BASE: float = 0.25
    AXES_PADDING: float = 0.2
    LABEL_PADDING: float = 0.2
LABEL = _Label_()


@dataclass(frozen=True)
class _Plotter_:
    FEATURE: float = '_feature_'
    TRANSFORMED_FEATURE: float = '_transformed_'
    F_BASE_NAME: str = '_base_'
    DEFAULT_F_BASE: int = 0
    ERR_LOW: str = '_error_lower_'
    ERR_UPP: str = '_error_upper_'
    RIDGE_SHIFT: float = -0.3
    KD_SEQUENCE_LEN: int = 300
    FITTED_VALUES_NAME: str = '_fitted_values_'
    FIT_CI_LOW: str = '_fit_ci_low_'
    FIT_CI_UPP: str = '_fit_ci_upp_'
    PRED_CI_LOW: str = '_pred_ci_low_'
    PRED_CI_UPP: str = '_pred_ci_upp_'
    FIT_LINE_ZORDER: float = 2.3
    FIT_CI_ZORDER: float = 2.2
    PRED_CI_ZORDER: float = 2.1
    @property
    def REGRESSION_CI_NAMES(self) -> Tuple[str]:
        """Get names for regression confidences in order
        - lower confidence level of fitted values
        - upper confidence level of fitted values
        - lower confidence level of predicted values
        - upper confidence level of predicted values"""
        names = (
            self.FIT_CI_LOW, self.FIT_CI_UPP,
            self.PRED_CI_LOW, self.PRED_CI_UPP)
        return names
PLOTTER = _Plotter_()


@dataclass(frozen=True)
class _Category_:
    MARKERS: Tuple[str] = ('o', 's', '^', 'p', 'D', 'v', 'P', 'X', '*')
    MARKERSIZE_LIMITS: Tuple[int] = (1, 13)
    N_SIZE_BINS: int = 5
    FEATURE_SPACE: float = 0.8
    FEATURE_PAD: float = 0.05
    @property
    def COLORS(self) -> Tuple[str]:
        return tuple(COLOR.PALETTE)
    @property
    def SIZE_LIMITS(self) -> Tuple[int]:
        """Used for scatter plots. The area must be specified there 
        instead of the height, as with markers in line plots.
        See: https://stackoverflow.com/a/14860958/11362192"""
        return tuple(s**2 for s in self.MARKERSIZE_LIMITS)
    @property
    def HANDLE_SIZES(self) -> Tuple[int]:
        """Get marker sizes for legend handles"""
        sizes = tuple(np.linspace(
            *self.MARKERSIZE_LIMITS, self.N_SIZE_BINS, dtype=int))
        return sizes
    @property
    def MARKER(self):
        """Get default marker if markers not specified"""
        return self.MARKERS[0]
CATEGORY = _Category_()


@dataclass(frozen=True)
class _Distribution_:
    _ignore_: Tuple[str] = ('levy_stable', 'studentized_range')
    COMMON: Tuple[str] = (
        'norm', 'chi2', 'foldnorm', 'rayleigh', 'weibull_min', 'gamma', 'wald',
        'expon', 'logistic', 'lognorm')
    @property
    def POSSIBLE(self) -> Tuple[str]:
        """Get all possible continous distributions coming from scipy"""
        return tuple(d for d in _distn_names if d not in self._ignore_)
DISTRIBUTION = _Distribution_()


__all__ = [
    'KW',
    'KDE',
    'COLOR',
    'LABEL',
    'PLOTTER',
    'CATEGORY',
    'DISTRIBUTION',
]