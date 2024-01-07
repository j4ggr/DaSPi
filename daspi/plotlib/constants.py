import matplotlib.pyplot as plt

from typing import List
from datetime import date

KDE_POINTS = 300
KDE_PADDING = 0.05

class _Kw_:
    @property
    def LEGEND(self) -> dict:
        """Key word arguments for figure legend"""
        return dict(loc='center right', bbox_to_anchor=(0.98, 0.5))
    @property
    def INFO(self) -> dict:
        """Key word arguments for adding info text"""
        return dict(x=0.02, y=-0.03, size='x-small')
KW = _Kw_()


class _Color_:
    @property
    def PALETTE(self) -> List[str]:
        """Get prop cycler color palette"""
        return plt.rcParams['axes.prop_cycle'].by_key()['color']
COLOR = _Color_()


__all__ = [
    'KDE_POINTS',
    'KDE_PADDING',
    KW.__name__,
    COLOR.__name__,
]