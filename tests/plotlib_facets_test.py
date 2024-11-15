import sys
import pytest

from pathlib import Path


sys.path.append(Path(__file__).parent.resolve()) # type: ignore


from daspi import AxesFacets


class TestFacets:
    axs = AxesFacets(
        2, 2, sharex='col', sharey='row', 
        width_ratios=[4, 1], height_ratios=[1, 4])
    # cat_axs = CategoricalAxesFacets()
    
    def test_iteration(self) -> None:
        assert self.axs.ax is None
        assert next(iter(self.axs)) == self.axs[0]
        for i, ax in enumerate(self.axs):
            assert self.axs.ax == ax
            assert self.axs[i] == ax

    def test_label_facets(self):...