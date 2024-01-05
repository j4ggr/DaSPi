import matplotlib.pyplot as plt

from os import environ
from typing import List
from datetime import date


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


class _String_:
    USERNAME: str = environ['USERNAME']
    @property
    def TODAY(self):
        return date.today().strftime('%Y.%m.%d')
STR = _String_()


class _Color_:
    @property
    def PALETTE(self) -> List[str]:
        """Get prop cycler color palette"""
        return plt.rcParams['axes.prop_cycle'].by_key()['color']
COLOR = _Color_()

__all__ = [
    KW.__name__,
    STR.__name__,
    COLOR.__name__,
]