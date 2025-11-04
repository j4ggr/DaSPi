import sys
import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from typing import Any
from typing import Generator
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi import JointChart
from daspi import SingleChart
from daspi import MultivariateChart
from daspi import SpecLimits
from daspi.plotlib.plotter import Scatter
from daspi.plotlib.appearance import positions_of_shared_axes

class TestSharingAxesFunctions:
    """tests for get_shared_axes and positions_of_shared_axes functions"""
    @pytest.fixture
    def shared_axes(self) -> Generator[NDArray, Any, None]:
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row')
        yield axs
        plt.close(fig)
    
    def test_position_types(self, shared_axes: NDArray) -> None:
        ax = shared_axes[0, 0]
        positions = positions_of_shared_axes(ax, 'x')
        assert isinstance(positions, list)
        assert all(isinstance(p, int) for p in positions)

        positions = positions_of_shared_axes(ax, 'y')
        assert isinstance(positions, list)
        assert all(isinstance(p, int) for p in positions)
    
    def test_axes_order(self, shared_axes: NDArray) -> None:
        figure_axes = shared_axes[0, 0].figure.axes
        assert figure_axes[0] == shared_axes[0, 0]
        assert figure_axes[1] == shared_axes[0, 1]
        assert figure_axes[2] == shared_axes[1, 0]
        assert figure_axes[3] == shared_axes[1, 1]

    def test_positions_shared_x_axis(self, shared_axes: NDArray) -> None:
        positions = positions_of_shared_axes(shared_axes[0, 0], 'x')
        assert positions == [0, 2]
        positions = positions_of_shared_axes(shared_axes[0, 1], 'x')
        assert positions == [1, 3]
        positions = positions_of_shared_axes(shared_axes[1, 0], 'x')
        assert positions == [0, 2]
        positions = positions_of_shared_axes(shared_axes[1, 1], 'x')
        assert positions == [1, 3]

    def test_positions_shared_y_axis(self, shared_axes: NDArray) -> None:
        positions = positions_of_shared_axes(shared_axes[0, 0], 'y')
        assert positions == [0, 1]
        positions = positions_of_shared_axes(shared_axes[0, 1], 'y')
        assert positions == [0, 1]
        positions = positions_of_shared_axes(shared_axes[1, 0], 'y')
        assert positions == [2, 3]
        positions = positions_of_shared_axes(shared_axes[1, 1], 'y')
        assert positions == [2, 3]

    def test_positions_no_shared_axes(self) -> None:
        fig, ax = plt.subplots(1, 1)
        positions = positions_of_shared_axes(ax, 'x')
        assert isinstance(positions, list)
        assert len(positions) == 1
        assert positions == [0]
        plt.close(fig)

    def test_positions_empty_figure(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        positions = positions_of_shared_axes(ax, 'x')
        assert isinstance(positions, list)
        assert len(positions) == 1
        plt.close(fig)


class TestSingleChart:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'feature': ['A', 'B', 'C', 'D', 'E']
        })

    def test_initialization(self, sample_data: pd.DataFrame) -> None:
        chart = SingleChart(sample_data, 'target', 'feature')
        assert chart.source.equals(sample_data)
        assert chart.target == 'target'
        assert chart.feature == 'feature'
        assert chart.target_on_y == True

    def test_initialization_target_on_x(self, sample_data: pd.DataFrame) -> None:
        chart = SingleChart(sample_data, 'target', 'feature', target_on_y=False)
        assert chart.target_on_y == False

    def test_initialization_with_kwds(self, sample_data: pd.DataFrame) -> None:
        chart = SingleChart(sample_data, 'target', 'feature', sharex='col', sharey='row')
        assert chart.axes.sharex == 'col'
        assert chart.axes.sharey == 'row'


class TestJointChart:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],
            'z': [1.1, 0.9, 1.0, 1.2, 0.8],
            'category': ['A', 'B', 'A', 'B', 'A'],
        })

    def test_initialization(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(sample_data, feature='x', target='y', ncols=1, nrows=1)
        assert isinstance(chart, JointChart)
        assert chart.source.equals(sample_data)
        assert chart.features == ('x',)
        assert chart.targets == ('y',)
        assert chart.target_on_ys == (True,)

    def test_hue(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3)
        assert chart.n_axes == 6
        assert chart.features == tuple(['x']*chart.n_axes)
        assert chart.targets == tuple(['y']*chart.n_axes)
        assert chart.hues == tuple(['category']*chart.n_axes)
    
    def test_normalize_to_tuple(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3)
        assert chart.n_axes == 6
        assert chart.normalize_to_tuple('x') == ('x', 'x', 'x', 'x', 'x', 'x')

        with pytest.raises(AssertionError) as err:
            chart.normalize_to_tuple(('x', 'x'))
            assert 'not enough values' in str(err)

    def test_itercharts(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3)
        axes = chart.axes.flat
        for i, _chart in enumerate(chart.itercharts()):
            assert _chart == chart.charts[i]
            assert chart.axes.ax == axes[i]
    
    def test_specification_limits_iterator(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3)
        
        spec_limits = SpecLimits(1.0, 2.0)
        limits = chart.specification_limits_iterator(spec_limits)
        assert isinstance(limits, Generator)
        for limit in limits:
            assert limit == spec_limits
        
        spec_limits = (
            SpecLimits(1.0, 2.0), SpecLimits(3.0, 4.0), SpecLimits(5.0, 6.0), 
            SpecLimits(7.0, 8.0), SpecLimits(9.0, 10.0), SpecLimits(11.0, 12.0))
        limits = chart.specification_limits_iterator(spec_limits)
        for i, limit in enumerate(limits):
            assert limit == spec_limits[i]

        spec_limits = (SpecLimits(1.0, 2.0), SpecLimits(3.0, 4.0))  # Less limits than n_axes
        with pytest.raises(AssertionError) as err:
            limits = chart.specification_limits_iterator(spec_limits)
            next(limits)

    def test_share_axis(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3, target_on_y=True, sharex=True, sharey='all')
        assert chart.axes_share_feature == True
        assert chart.axes_share_feature == chart.axes.sharex
        assert chart.axes_share_target == 'all'
        assert chart.axes_share_target == chart.axes.sharey

        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3, target_on_y=False, sharex='row', sharey='none')
        assert chart.axes_share_feature == 'none'
        assert chart.axes_share_feature == chart.axes.sharey
        assert chart.axes_share_target == 'row'
        assert chart.axes_share_target == chart.axes.sharex

        with pytest.warns(UserWarning):
            chart = JointChart(
                sample_data, feature='x', target='y', hue='category',
                ncols=2, nrows=1, target_on_y=(True, False), sharex=True)

    def test_single_label_allowed(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3, sharex='none', sharey='none')
        assert chart.single_label_allowed(is_target=False)
        assert chart.single_label_allowed(is_target=True)

        chart = JointChart(
            sample_data, feature=('x', 'category'), target=('y', 'z'), 
            ncols=2, nrows=1, sharex=True, sharey='none')
        assert chart.single_label_allowed(is_target=False)
        assert not chart.single_label_allowed(is_target=True)

        chart = JointChart(
            sample_data, feature=('x', 'category'), target=('y', 'z'), 
            ncols=2, nrows=1, sharex=True, sharey='none', target_on_y=False)
        assert not chart.single_label_allowed(is_target=False)
        assert chart.single_label_allowed(is_target=True)


class TestChartFormatting:
    """Test the new formatting features for SingleChart and MultivariateChart."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'x': np.random.normal(0, 1, 30),
            'y': np.random.normal(0, 1, 30),
            'category': np.random.choice(['A', 'B', 'C'], 30),
            'facility': np.random.choice(['North', 'South'], 30)
        })

    def test_single_chart_parameter_mapping_standard(self, sample_data: pd.DataFrame) -> None:
        """Test parameter mapping for SingleChart with target_on_y=True."""
        chart = SingleChart(
            source=sample_data,
            target='y',
            feature='x',
            target_on_y=True
        )
        
        # Test standard orientation mapping
        result = chart._map_axis_parameters(
            feature_formatter=lambda x: f"{x:.1f}",
            target_formatter=lambda x: f"{x:.2f}",
            feature_angle=30,
            target_angle=45,
            feature_align='left',
            target_align='right'
        )
        
        xlabel_formatter, ylabel_formatter, xlabel_angle, ylabel_angle, xlabel_align, ylabel_align = result
        
        # With target_on_y=True: feature->x, target->y
        assert xlabel_formatter is not None  # feature formatter goes to x
        assert ylabel_formatter is not None  # target formatter goes to y
        assert xlabel_angle == 30  # feature angle goes to x
        assert ylabel_angle == 45  # target angle goes to y
        assert xlabel_align == 'left'  # feature align goes to x (unchanged)
        assert ylabel_align == 'top'  # target align 'right' maps to 'top' for y-axis

    def test_single_chart_parameter_mapping_swapped(self, sample_data: pd.DataFrame) -> None:
        """Test parameter mapping for SingleChart with target_on_y=False."""
        chart = SingleChart(
            source=sample_data,
            target='y',
            feature='x',
            target_on_y=False
        )
        
        # Test swapped orientation mapping
        result = chart._map_axis_parameters(
            feature_formatter=lambda x: f"{x:.1f}",
            target_formatter=lambda x: f"{x:.2f}",
            feature_angle=30,
            target_angle=45,
            feature_align='left',
            target_align='right'
        )
        
        xlabel_formatter, ylabel_formatter, xlabel_angle, ylabel_angle, xlabel_align, ylabel_align = result
        
        # With target_on_y=False: target->x, feature->y
        assert xlabel_formatter is not None  # target formatter goes to x
        assert ylabel_formatter is not None  # feature formatter goes to y
        assert xlabel_angle == 45  # target angle goes to x
        assert ylabel_angle == 30  # feature angle goes to y
        assert xlabel_align == 'right'  # target align goes to x (unchanged)
        assert ylabel_align == 'bottom'  # feature align 'left' maps to 'bottom' for y-axis

    def test_single_chart_alignment_mapping(self, sample_data: pd.DataFrame) -> None:
        """Test alignment mapping from general to axis-specific alignments."""
        chart = SingleChart(
            source=sample_data,
            target='y',
            feature='x'
        )
        
        # Test all alignment combinations
        test_cases = [
            ('left', 'center', 'left', 'center'),
            ('center', 'left', 'center', 'bottom'),
            ('right', 'right', 'right', 'top')
        ]
        
        for feature_align, target_align, expected_x, expected_y in test_cases:
            result = chart._map_axis_parameters(
                feature_align=feature_align,  # type: ignore
                target_align=target_align  # type: ignore
            )
            
            _, _, _, _, xlabel_align, ylabel_align = result
            assert xlabel_align == expected_x
            assert ylabel_align == expected_y

    def test_single_chart_formatter_integration(self, sample_data: pd.DataFrame) -> None:
        """Test that formatters are properly integrated into the chart."""
        def custom_formatter(x):
            return f"Value: {x:.1f}"
        
        chart = SingleChart(
            source=sample_data,
            target='y',
            feature='x'
        ).plot(
            Scatter,
            alpha=0.5
        ).label(
            target_formatter=custom_formatter,
            feature_formatter=lambda x: f"X: {x:.2f}"
        )
        
        # Verify that the chart was created successfully
        assert hasattr(chart, 'label_facets')
        assert chart.label_facets is not None

    def test_multivariate_chart_parameter_integration(self, sample_data: pd.DataFrame) -> None:
        """Test that MultivariateChart properly integrates formatting parameters."""
        def temp_formatter(x):
            return f"{x:.1f}°C"
        
        chart = MultivariateChart(
            source=sample_data,
            target='y',
            feature='x',
            col='facility'
        ).plot(
            Scatter,
            alpha=0.5
        ).label(
            target_formatter=temp_formatter,
            feature_angle=25,
            target_align='right'
        )
        
        # Verify that the chart was created successfully
        assert hasattr(chart, 'label_facets')
        assert chart.label_facets is not None

    def test_chart_with_all_formatting_options(self, sample_data: pd.DataFrame) -> None:
        """Test chart with all formatting options combined."""
        chart = SingleChart(
            source=sample_data,
            target='y',
            feature='category',
            categorical_feature=True,
            hue='facility'
        ).plot(
            Scatter,
            alpha=0.7
        ).label(
            fig_title='Test Chart with All Options',
            target_label='Target Variable',
            feature_label='Feature Variable',
            target_formatter=lambda x: f"{x:.2f} units",
            feature_formatter=lambda x: f"Cat {x}",
            target_angle=15,
            feature_angle=45,
            target_align='center',
            feature_align='right'
        )
        
        # Verify chart creation and label facets
        assert hasattr(chart, 'label_facets')
        assert chart.label_facets.xlabel_angle == 45
        assert chart.label_facets.ylabel_angle == 15
