import sys
import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from matplotlib.axis import YAxis
from matplotlib.figure import Figure

from typing import Any
from typing import Generator
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi import JointChart
from daspi import SingleChart
from daspi.plotlib.appearance import get_shared_axes
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
    
    def test_ensure_tuple(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3)
        assert chart.n_axes == 6
        assert chart.ensure_tuple('x') == tuple(['x']*chart.n_axes)

        with pytest.raises(AssertionError) as err:
            chart.ensure_tuple(('x', 'x'))
            assert 'not enough values' in str(err)
        
        with pytest.raises(ValueError) as err:
            chart.ensure_tuple(1.0) # type: ignore

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
        
        spec_limits = (1.0, 2.0)
        limits = chart.specification_limits_iterator(spec_limits)
        assert isinstance(limits, Generator)
        for limit in limits:
            assert limit == spec_limits
        
        spec_limits = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0), (9.0, 10.0), (11.0, 12.0))
        limits = chart.specification_limits_iterator(spec_limits)
        for i, limit in enumerate(limits):
            assert limit == spec_limits[i]

        spec_limits = ((1.0, 2.0), (3.0, 4.0))  # Less limits than n_axes
        limits = chart.specification_limits_iterator(spec_limits)
        with pytest.raises(AssertionError) as err:
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

