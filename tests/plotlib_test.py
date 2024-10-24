import sys
import pytest

import numpy as np
import pandas as pd

from pytest import approx
from typing import Generator
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve())

from daspi import CATEGORY
from daspi import Dodger
from daspi import HueLabel
from daspi import SizeLabel
from daspi import JointChart
from daspi import ShapeLabel
from daspi import AxesFacets
from daspi import SingleChart
from daspi import BivariateUnivariateCharts


class TestCategoryLabel:
    colors = HueLabel(('alpha', 'beta', 'gamma', 'delta'), CATEGORY.PALETTE)
    markers = ShapeLabel(('foo', 'bar', 'bazz'), CATEGORY.MARKERS)
    sizes = SizeLabel(1.5, 3.5, CATEGORY.N_SIZE_BINS)

    def test_str(self) -> None:
        assert str(self.colors) == 'HueLabel'
        assert str(self.markers) == 'ShapeLabel'
        assert str(self.sizes) == 'SizeLabel'

    def test_errrors(self) -> None:
        with pytest.raises(AssertionError) as err:
            self.colors['psi']
        assert "Can't get category for label" in str(err.value)
        
        n = self.colors.n_allowed
        with pytest.raises(AssertionError) as err:
            HueLabel([i for i in range(n+1)], CATEGORY.PALETTE)
        assert str(err.value) == (
            f'HueLabel can handle {n} categories, got {n+1}')

        with pytest.raises(AssertionError) as err:
            ShapeLabel(('foo', 'foo', 'bar', 'bazz'), CATEGORY.MARKERS)
        assert str(err.value) == (
            'Labels occur more than once, only unique labels are allowed')

    def test_handles_labels(self) -> None:
        handles, labels = self.colors.handles_labels()
        assert len(self.colors.colors) == self.colors.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.markers.handles_labels()
        assert len(self.markers.markers) == self.markers.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.sizes.handles_labels()
        assert len(self.sizes.categories) == self.sizes.n_used
        assert len(handles) == len(labels)
        assert len(handles) == CATEGORY.N_SIZE_BINS

    def test_getitem(self) -> None:
        assert self.colors['alpha'] == CATEGORY.PALETTE[0]
        assert self.colors['beta'] == CATEGORY.PALETTE[1]
        assert self.colors['gamma'] == CATEGORY.PALETTE[2]
        assert self.colors['delta'] == CATEGORY.PALETTE[3]

        assert self.markers['foo'] == CATEGORY.MARKERS[0]
        assert self.markers['bar'] == CATEGORY.MARKERS[1]
        assert self.markers['bazz'] == CATEGORY.MARKERS[2]
    
    def test_sizes(self) -> None:
        s_min, s_max = CATEGORY.MARKERSIZE_LIMITS
        assert self.sizes[1.5] == s_min**2
        assert self.sizes[3.5] == s_max**2


        values = np.linspace(
            self.sizes._min, self.sizes._max, CATEGORY.N_SIZE_BINS)
        sizes = self.sizes(values)
        assert np.array_equal(sizes, np.square(self.sizes.categories))


class TestDodger:

    def test_getitem(self) -> None:
        tick_labels = ('a', 'b', 'c', 'd')
        dodge = Dodger(('r', ), tick_labels=tick_labels)
        assert dodge['r'] == 0

        dodge = Dodger(('r', 'g'), tick_labels=tick_labels)
        assert dodge['r'] == approx(-0.2)
        assert dodge['g'] == approx(0.2)

        dodge = Dodger(('r', 'g', 'b', 'y'), tick_labels=tick_labels)
        assert dodge['r'] == approx(-0.3)
        assert dodge['g'] == approx(-0.1)
        assert dodge['b'] == approx(0.1)
        assert dodge['y'] == approx(0.3)
        
        dodge = Dodger((), tick_labels=tick_labels)
        assert bool(dodge.dodge) == False
        assert bool(dodge['r']) == dodge._default
        
        dodge = Dodger((), tick_labels=())
        assert bool(dodge.dodge) == False
        assert bool(dodge['r']) == dodge._default

    def test_call(self) -> None:
        tick_labels = ('a', 'b', 'c', 'd')
        categories = ('r', 'g', 'b', 'y')
        dodge = Dodger(categories=categories, tick_labels=tick_labels)
        old = pd.Series(tick_labels)
        assert dodge(old, 'r').values == approx(dodge.ticks - 0.3)
        assert dodge(old, 'g').values == approx(dodge.ticks - 0.1)
        assert dodge(old, 'b').values == approx(dodge.ticks + 0.1)
        assert dodge(old, 'y').values == approx(dodge.ticks + 0.3)


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
        assert chart.axes_facets.sharex == 'col'
        assert chart.axes_facets.sharey == 'row'


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
            chart.ensure_tuple(1.0)

    def test_itercharts(self, sample_data: pd.DataFrame) -> None:
        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3)
        axes = chart.axes.flatten()
        for i, _chart in enumerate(chart.itercharts()):
            assert _chart == chart.charts[i]
            assert chart.axes_facets.ax == axes[i]
    
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
        assert chart.axes_share_feature == chart.axes_facets.sharex
        assert chart.axes_share_target == 'all'
        assert chart.axes_share_target == chart.axes_facets.sharey

        chart = JointChart(
            sample_data, feature='x', target='y', hue='category',
            ncols=2, nrows=3, target_on_y=False, sharex='row', sharey='none')
        assert chart.axes_share_feature == 'none'
        assert chart.axes_share_feature == chart.axes_facets.sharey
        assert chart.axes_share_target == 'row'
        assert chart.axes_share_target == chart.axes_facets.sharex

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


class TestBivariateUnivariateChart:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        cat = ['A', 'A', 'B', 'B', 'A', 'B']
        return pd.DataFrame({
            'x': [11.5, 8.5, 9, 11, 8, 10.5]*3,
            'z': [1.1, 0.9, 1.0, 1.2, 0.8, 1.0]*3,
            'category': cat*2 + cat[::-1],
        })
    
    def test_init(self, sample_data: pd.DataFrame) -> None:
        chart = BivariateUnivariateCharts(
            sample_data,
            feature='x',
            target='y',
            hue='category')
        assert chart.features == ('', '', 'x', '')
        assert chart.targets == ('x', '', 'y', 'y')
        assert chart.hues == ('category', 'category', 'category', 'category')
        assert chart.target_on_ys == (False, True, True, True)
        assert chart.n_axes == 4
        assert chart.dodges == tuple([False]*4)
        assert chart.categorical_features == tuple([False]*4)

        chart = BivariateUnivariateCharts(
            sample_data,
            feature='x',
            target='y',
            hue='category',
            dodge_univariates=True)
        assert chart.n_axes == 4
        assert chart.features == ('', '', 'x', '')
        assert chart.targets == ('x', '', 'y', 'y')
        assert chart.hues == ('category', 'category', 'category', 'category')
        assert chart.target_on_ys == (False, True, True, True)
        assert chart.dodges == (True, False, False, True)
        assert chart.categorical_features == chart.dodges

