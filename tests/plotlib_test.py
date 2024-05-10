import sys
import pytest

import numpy as np
import pandas as pd

from pytest import approx
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve())

from daspi.constants import DEFAULT
from daspi.constants import CATEGORY
from daspi.plotlib.classify import Dodger
from daspi.plotlib.classify import HueLabel
from daspi.plotlib.classify import SizeLabel
from daspi.plotlib.classify import ShapeLabel
from daspi.plotlib.facets import AxesFacets


class TestCategoryLabel:
    colors = HueLabel(('alpha', 'beta', 'gamma', 'delta'))
    markers = ShapeLabel(('foo', 'bar', 'bazz'))
    sizes = SizeLabel(1.5, 3.5)

    def test_str(self):
        assert str(self.colors) == 'HueLabel'
        assert str(self.markers) == 'ShapeLabel'
        assert str(self.sizes) == 'SizeLabel'

    def test_errrors(self):
        with pytest.raises(AssertionError) as err:
            self.colors['psi']
        assert "Can't get category for label" in str(err.value)
        
        with pytest.raises(AssertionError) as err:
            n = self.colors.n_allowed
            HueLabel([i for i in range(n+1)])
        assert str(err.value) == (
            f'HueLabel can handle {n} categories, got {n+1}')

        with pytest.raises(AssertionError) as err:
            ShapeLabel(('foo', 'foo', 'bar', 'bazz'))
        assert str(err.value) == (
            'Labels occur more than once, only unique labels are allowed')

    def test_handles_labels(self):
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

    def test_getitem(self):
        assert self.colors['alpha'] == CATEGORY.PALETTE[0]
        assert self.colors['beta'] == CATEGORY.PALETTE[1]
        assert self.colors['gamma'] == CATEGORY.PALETTE[2]
        assert self.colors['delta'] == CATEGORY.PALETTE[3]

        assert self.markers['foo'] == CATEGORY.MARKERS[0]
        assert self.markers['bar'] == CATEGORY.MARKERS[1]
        assert self.markers['bazz'] == CATEGORY.MARKERS[2]
    
    def test_sizes(self):
        s_min, s_max = CATEGORY.MARKERSIZE_LIMITS
        assert self.sizes[1.5] == s_min**2
        assert self.sizes[3.5] == s_max**2


        values = np.linspace(
            self.sizes._min, self.sizes._max, CATEGORY.N_SIZE_BINS)
        sizes = self.sizes(values)
        assert np.array_equal(sizes, np.square(self.sizes.categories))


class TestDodger:

    def test_getitem(self):
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

    def test_call(self):
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
    
    def test_iteration(self):
        assert self.axs.ax is None
        assert next(iter(self.axs)) == self.axs[0]
        for i, ax in enumerate(self.axs):
            assert self.axs.ax == ax
            assert self.axs[i] == ax

    def test_label_facets(self):...


