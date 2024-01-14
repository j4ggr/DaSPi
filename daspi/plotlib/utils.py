import numpy as np

from typing import Any
from typing import List
from typing import Tuple
from typing import Literal
from numpy.typing import ArrayLike

from matplotlib.axes import Axes

from .._constants import CATEGORY


def add_second_axis(# TODO remove if not needed
        ax: Axes, which: Literal['x', 'y'], tick_labels: List[str], 
        axis_label: str = '') -> Axes:
    """Adds e second axis to the given direction, sharing the other one.
    
    Parameters
    ----------
    ax : Axes
        Base axes object to add second axis
    which : {'x', 'y'}
        From which axis a second one should be added
    tick_labels : list of str, optional
        Axis tick labels
    axis_label : str, optional
        Label for axis, corresponds to ylabel if which is 'y', else to
        xlabel, by default ''
    
    Returns
    -------
    ax2 : Axes
        The added axis object
    """
    ticks = [i for i in range(len(tick_labels))]
    ax2 = ax.twinx() if which == 'y' else ax.twiny()
    keys = (f'{which}{l}' for l in ('label', 'ticks', 'ticklabels'))
    values = (axis_label, ticks, tick_labels)
    ax2.set(**{k: v for k, v in zip(keys, values)})
    
    margins = ax.margins()
    ax2.margins(x=margins[0]/2, y=margins[1]/2)
    return ax2


class BaseCategoryLabelHandler:

    __slots__ = ('_categories', '_labels')
    _categories: Tuple[str]
    _labels: Tuple[str]
    
    def __init__(self, labels: Tuple) -> None:
        self._labels: Tuple = ()
        self.labels = labels

    @property
    def categories(self) -> Tuple[str]:
        return self._categories

    @property
    def labels(self) -> Tuple:
        return self._labels
    @labels.setter
    def labels(self, labels: Tuple):
        assert len(labels) <= self.n_allowed, f'{self} can handle {self.n_allowed} categories, got {len(labels)}'
        assert len(labels) == len(set(labels)), f'One or more labels occur more than once, only unique labels are allowed'
        self._labels = labels

    @property
    def n_allowed(self) -> int:
        """Allowed amount of categories"""
        return len(self.categories)
    
    def __getitem__(self, label: Any) -> str:
        try:
            idx = self.labels.index(label)
        except ValueError:
            raise KeyError(f"Can't get category for label {label}, got {self.labels}")

        return self.categories[idx]

    def __str__(self) -> str:
        return self.__class__.__name__
    

class HueLabelHandler(BaseCategoryLabelHandler):

    _categories = CATEGORY.COLORS

    def __init__(self, labels: Tuple) -> None:
        super().__init__(labels)


class MarkerLabelHandler(BaseCategoryLabelHandler):

    _categories = CATEGORY.MARKERS

    def __init__(self, labels: Tuple) -> None:
        super().__init__(labels)

__all__ = [
    add_second_axis.__name__,
    ]