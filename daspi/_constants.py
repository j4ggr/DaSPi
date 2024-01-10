import matplotlib.pyplot as plt

from typing import Tuple
from typing import List
from dataclasses import dataclass
from scipy.stats._continuous_distns import _distn_names


class _Kw_:
    @property
    def LINE(self):
        """Base kwargs for horizontal or vertical lines"""
        return dict(ls=0.8, style='--')
    @property
    def LEGEND(self) -> dict:
        """Figure legend at right side of figure"""
        return dict(loc='top right', bbox_to_anchor=(0.98, 0.5))
    @property
    def INFO(self) -> dict:
        """Adding info text at bottom left of figure"""
        return dict(x=0.02, y=-0.03, size='x-small')
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
    MEAN: str = '#000000'
    PERCENTIL: str = '#303030'
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
class _Distribution_:
    _ignore_: Tuple[str] = ('levy_stable', 'studentized_range')
    COMMON: Tuple[str] = ('norm', 'weibull_min', 'lognorm', 'expon')
    @property
    def POSSIBLE(self) -> Tuple[str]:
        """Get all possible continous distributions coming from scipy"""
        return tuple(d for d in _distn_names if d not in self._ignore_)
DISTRIBUTION = _Distribution_()


__all__ = [
    'KW',
    'KDE',
    'COLOR',
    'DISTRIBUTION',
]