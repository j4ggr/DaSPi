import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from typing import List
from dataclasses import dataclass
from scipy.stats._continuous_distns import _distn_names


class _Kw_:
    @property
    def LINE(self):
        """Base kwds for horizontal or vertical lines"""
        return dict(ls=0.8, style='--')
    @property
    def LEGEND(self) -> dict:
        """Figure legend at right side of figure"""
        return dict(loc='upper left', bbox_to_anchor=(1.2, 1), alignment='left')
    @property
    def SAVE_CHART(self) -> dict:
        """Key word arguments for matplotlib savefig"""
        return dict(bbox_inches='tight')
    @property
    def XLABEL(self) -> dict:
        """Keyword arguments for xlabel when using figure.text method.
        This is the case if the diagram has several columns"""
        return dict(
            x=0.5, y=0, ha='center', va='top')
    @property
    def YLABEL(self) -> dict:
        """Keyword arguments for ylabel when using figure.text method.
        This is the case if the diagram has several rows"""
        return dict(
            x=0, y=0.5, ha='right', va='center', rotation='vertical')
    @property
    def ROW_LABEL(self) -> dict:
        """Keyword arguments for Axes.text method used for adding column 
        label as text at LabelFacets"""
        return dict(x=1, y=0.5, ha='left', va='center', rotation=-90)
    @property
    def ROW_TITLE(self) -> dict:
        """Keyword arguments for Axes.text method used for adding column 
        label as text at LabelFacets"""
        return dict(x=1.1, y=0.5, ha='left', va='center', rotation=-90)
    @property
    def COL_LABEL(self) -> dict:
        """Keyword arguments for Axes.text method used for adding row 
        label as text at LabelFacets"""
        return dict(x=0.5, y=1, ha='center', va='bottom')
    @property
    def COL_TITLE(self) -> dict:
        """Keyword arguments for Axes.text method used for adding row 
        label as text at LabelFacets centrally over the middle axis"""
        return dict(x=0.5, y=1.1, ha='center', va='bottom')
    @property
    def SUB_TITLE(self) -> dict:
        """Keyword arguments for Axes.set_title method used for adding
        sub title at LabelFacets"""
        return dict(loc='left')
    @property
    def INFO(self) -> dict:
        """Adding info text at bottom left of figure"""
        return dict(x=0.02, y=0, size='x-small')
KW = _Kw_()


@dataclass(frozen=True)
class _Kde_:
    POINTS: int = 300
    PADDING: float = 0.5
KDE = _Kde_()


@dataclass(frozen=True)
class _Color_:
    GOOD: str = '#2ca02c'
    BAD: str = '#d62728'
    MEAN: str = '#101010'
    PERCENTIL: str = '#303030'
    HANDLES: str = '#202020'
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
class _Category_:
    COLORS: Tuple[str] = tuple(COLOR.PALETTE)
    MARKERS: Tuple[str] = ('o', 's', '^', 'p', 'D', 'v', 'P', 'X', '*')
    MARKERSIZE_LIMITS: Tuple[int] = (1, 13)
    N_SIZE_BINS: int = 5
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
    'CATEGORY',
    'DISTRIBUTION',
]