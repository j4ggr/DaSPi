import sys
import pytest
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter


sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi import AxesFacets
from daspi import flat_unique
from daspi.plotlib.facets import LabelFacets


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
            nrows=2, ncols=2, sharex='col', sharey='row', 
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
            ['Horizontal', 'Horizontal', '.'],
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
        assert l_mosaic[0, -1] is None
        assert len(l_mosaic) == 3
        assert all(l_mosaic[0] == a for a in l_mosaic[0, :2])
        assert l_mosaic[0] != l_mosaic[0, -1]
        assert l_mosaic[0] not in l_mosaic[1:, :].flat
        assert all(l_mosaic[1] == a for a in l_mosaic[1:, 0])
        assert l_mosaic[1] not in l_mosaic[1:, 1:].flat
        assert all(l_mosaic[2] == a for a in l_mosaic[1:, 1:].flat)

    def test_iteration(self, l_mosaic: AxesFacets) -> None:
        assert l_mosaic.ax is None
        assert next(iter(l_mosaic)) == l_mosaic[0]
        for i, ax in enumerate(l_mosaic):
            assert l_mosaic.ax == ax
            assert ax == l_mosaic[i]


class TestLabelFacets:
    
    @pytest.fixture
    def simple_axes(self) -> AxesFacets:
        return AxesFacets(nrows=1, ncols=2)
    
    def percentage_formatter(self, x, pos):
        """Format tick values as percentages."""
        return f'{x:.1f}%'
    
    def test_label_facets_initialization(self, simple_axes: AxesFacets) -> None:
        """Test that LabelFacets initializes correctly with new parameters."""
        label_facets = LabelFacets(
            simple_axes,
            xlabel='Test X Label',
            ylabel='Test Y Label',
            xlabel_formatter=FuncFormatter(self.percentage_formatter),
            ylabel_formatter=FuncFormatter(self.percentage_formatter),
            xlabel_angle=45,
            ylabel_angle=30
        )
        
        assert label_facets.xlabel == 'Test X Label'
        assert label_facets.ylabel == 'Test Y Label'
        assert label_facets.xlabel_formatter is not None
        assert label_facets.ylabel_formatter is not None
        assert label_facets.xlabel_angle == 45
        assert label_facets.ylabel_angle == 30
    
    def test_label_facets_default_values(self, simple_axes: AxesFacets) -> None:
        """Test that LabelFacets has correct default values for new parameters."""
        label_facets = LabelFacets(simple_axes)
        
        assert label_facets.xlabel_formatter is None
        assert label_facets.ylabel_formatter is None
        assert label_facets.xlabel_angle == 0
        assert label_facets.ylabel_angle == 0
    
    def test_label_facets_draw_with_formatters_and_angles(self, simple_axes: AxesFacets) -> None:
        """Test that formatters and angles are applied correctly when drawing."""
        # Add some data to the axes
        for ax in simple_axes:
            ax.plot([1, 2, 3], [10, 20, 30])
        
        label_facets = LabelFacets(
            simple_axes,
            xlabel_formatter=FuncFormatter(self.percentage_formatter),
            ylabel_formatter=FuncFormatter(self.percentage_formatter),
            xlabel_angle=45,
            ylabel_angle=15
        )
        
        # Draw the labels
        label_facets.draw()
        
        # Verify that angles were applied
        for ax in simple_axes:
            x_tick_labels = ax.get_xticklabels()
            y_tick_labels = ax.get_yticklabels()
            
            if x_tick_labels:
                assert x_tick_labels[0].get_rotation() == 45
            if y_tick_labels:
                assert y_tick_labels[0].get_rotation() == 15
    
    def test_label_facets_alignment_features(self, simple_axes: AxesFacets) -> None:
        """Test that alignment features work correctly."""
        # Add some data to the axes
        for ax in simple_axes:
            ax.plot([1, 2, 3], [10, 20, 30])
        
        label_facets = LabelFacets(
            simple_axes,
            xlabel_align='left',
            ylabel_align='top',
            xlabel_angle=30,
            ylabel_angle=45
        )
        
        # Draw the labels
        label_facets.draw()
        
        # Verify alignment was applied
        for ax in simple_axes:
            x_tick_labels = ax.get_xticklabels()
            y_tick_labels = ax.get_yticklabels()
            
            if x_tick_labels:
                assert x_tick_labels[0].get_horizontalalignment() == 'left'
                assert x_tick_labels[0].get_rotation() == 30
            if y_tick_labels:
                assert y_tick_labels[0].get_verticalalignment() == 'top'
                assert y_tick_labels[0].get_rotation() == 45
    
    def test_label_facets_alignment_with_default_values(self, simple_axes: AxesFacets) -> None:
        """Test alignment with default center values."""
        # Add some data to the axes
        for ax in simple_axes:
            ax.plot([1, 2, 3], [10, 20, 30])
        
        label_facets = LabelFacets(simple_axes)
        
        # Verify default alignment values
        assert label_facets.xlabel_align == 'center'
        assert label_facets.ylabel_align == 'center'
        
        # Draw and verify no changes to alignment when center (default matplotlib behavior)
        label_facets.draw()
