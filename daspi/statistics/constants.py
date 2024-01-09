from typing import Tuple
from dataclasses import dataclass
from scipy.stats._continuous_distns import _distn_names
from scipy.stats._distn_infrastructure import rv_continuous
from scipy.stats import kurtosistest

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
    'DISTRIBUTION',
]