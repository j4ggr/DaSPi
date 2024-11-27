import sys
import pytest
import numpy as np
from pathlib import Path


sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi import AxesFacets
from daspi import flat_unique


def test_flat_unique_numpy() -> None:
    nested = np.array([[1, 1], [2, 3]])
    result = flat_unique(nested)
    assert result == [1, 2, 3]

    nested = np.array([[1, 1, 1], [2, 2, 3], [2, 2, 3]])
    result = flat_unique(nested)
    assert result == [1, 2, 3]

    nested = np.array([[2, 2, 3], [2, 2, 3], [1, 1, 1]])
    result = flat_unique(nested)
    assert result == [2, 3, 1]

def test_flat_unique_list() -> None:
    nested = [[1, 1], [2, 3]]
    result = flat_unique(nested)
    assert result == [1, 2, 3]

    nested = [[1, 1, 1], [2, 2, 3], [2, 2, 3]]
    result = flat_unique(nested)
    assert result == [1, 2, 3]

    nested = [[2, 2, 3], [2, 2, 3], [1, 1, 1]]
    result = flat_unique(nested)
    assert result == [2, 3, 1]

def test_flat_unique_empty() -> None:
    result = flat_unique([])
    assert result == []


class TestAxesFacets:

    @pytest.fixture
    def rc(self) -> AxesFacets:
        axes = AxesFacets(
            2, 2, sharex='col', sharey='row', 
            width_ratios=[4, 1], height_ratios=[1, 4])
        return axes
    
    @pytest.fixture
    def s_mosaic(self) -> AxesFacets:
        mosaic="""
            AAA
            BBC
            BBC"""
        axes = AxesFacets(
            mosaic=mosaic, sharex='col', sharey='row', 
            width_ratios=[2, 2, 1], height_ratios=[1, 2, 2])
        return axes
    
    @pytest.fixture
    def l_mosaic(self) -> AxesFacets:
        mosaic=[
            ['Horizontal', 'Horizontal', 'Qubic small'],
            ['Vertical', 'Qubic big', 'Qubic big'],
            ['Vertical', 'Qubic big', 'Qubic big']]
        axes = AxesFacets(
            mosaic=mosaic, sharex='col', sharey='row', 
            width_ratios=[2, 2, 1], height_ratios=[1, 2, 2])
        return axes
    
    def test_axes_grid_rc(self,rc: AxesFacets) -> None:
        assert rc.shape == (2, 2)
        assert rc.nrows == 2
        assert rc.ncols == 2
        assert len(rc) == 4

    def test_axes_grid_string(self, s_mosaic: AxesFacets) -> None:
        assert s_mosaic.shape == (3, 3)
        assert s_mosaic.nrows == 3
        assert s_mosaic.ncols == 3
        assert len(s_mosaic) == 3
        assert all(s_mosaic[0] == a for a in s_mosaic[0, :])
        assert not any(s_mosaic[0] == a for a in s_mosaic[1:, :].flat)
        assert all(s_mosaic[1] == a for a in s_mosaic[1:, :2].flat)
        assert not any(s_mosaic[1] == a for a in s_mosaic[1:, 2])
        assert all(s_mosaic[2] == a for a in s_mosaic[1:, 2])

    def test_axes_grid_lists(self, l_mosaic: AxesFacets) -> None:
        assert l_mosaic.shape == (3, 3)
        assert l_mosaic.nrows == 3
        assert l_mosaic.ncols == 3
        assert len(l_mosaic) == 4
        assert all(l_mosaic[0] == a for a in l_mosaic[0, :2])
        assert l_mosaic[0] != l_mosaic[0, -1]
        assert l_mosaic[0] not in l_mosaic[1:, :].flat
        assert all(l_mosaic[2] == a for a in l_mosaic[1:, 0])
        assert l_mosaic[2] not in l_mosaic[1:, 1:].flat
        assert all(l_mosaic[3] == a for a in l_mosaic[1:, 1:].flat)

    def test_iteration(self, l_mosaic: AxesFacets) -> None:
        assert l_mosaic.ax is None
        assert next(iter(l_mosaic)) == l_mosaic[0]
        for i, ax in enumerate(l_mosaic):
            assert l_mosaic.ax == ax
            assert ax == l_mosaic[i]
