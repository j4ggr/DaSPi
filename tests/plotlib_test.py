import sys
import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pytest import approx
from pathlib import Path
from numpy.typing import ArrayLike

sys.path.append(Path(__file__).parent.resolve())

from daspi._constants import CATEGORY
from daspi.plotlib.utils import HueLabelHandler
from daspi.plotlib.utils import SizeLabelHandler
from daspi.plotlib.utils import ShapeLabelHandler
from daspi.plotlib.chart import MultipleVariateChart
from daspi.plotlib.facets import AxesFacets
from daspi.plotlib.facets import CategoricalAxesFacets
from daspi.plotlib.plotter import Scatter

savedir = Path(__file__).parent/'charts'
savedir.mkdir(parents=True, exist_ok=True)
affairs = sm.datasets.fair.load_pandas()


class TestCategoryLabelHandler:
    colors = HueLabelHandler(('alpha', 'beta', 'gamma'))
    markers = ShapeLabelHandler(('foo', 'bar', 'bazz'))
    sizes_s = SizeLabelHandler(1.5, 3.5, 'scatter')
    sizes_l = SizeLabelHandler(1.5, 3.5, 'line')

    def test_str(self):
        assert str(self.colors) == 'HueLabelHandler'
        assert str(self.markers) == 'ShapeLabelHandler'
        assert str(self.sizes_s) == 'SizeLabelHandler'

    def test_errrors(self):
        with pytest.raises(KeyError) as err:
            self.colors['psi']
        assert str(err.value) == f"\"Can't get category for label 'psi', got {self.colors.labels}\""
        
        with pytest.raises(AssertionError) as err:
            n = self.colors.n_allowed
            HueLabelHandler([i for i in range(n+1)])
        assert str(err.value) == f'HueLabelHandler can handle {n} categories, got {n+1}'

        with pytest.raises(AssertionError) as err:
            ShapeLabelHandler(('foo', 'foo', 'bar', 'bazz'))
        assert str(err.value) == f'One or more labels occur more than once, only unique labels are allowed'

    def test_handles_labels(self):
        handles, labels = self.colors.handles_labels()
        assert len(self.colors.colors) == self.colors.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.markers.handles_labels()
        assert len(self.markers.markers) == self.markers.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.sizes_s.handles_labels()
        assert len(self.sizes_s.categories) == self.sizes_s.n_used
        assert len(handles) == len(labels)
        assert len(handles) == CATEGORY.N_SIZE_BINS

    def test_getitem(self):
        assert self.colors['alpha'] == CATEGORY.COLORS[0]
        assert self.colors['beta'] == CATEGORY.COLORS[1]
        assert self.colors['gamma'] == CATEGORY.COLORS[2]

        assert self.markers['foo'] == CATEGORY.MARKERS[0]
        assert self.markers['bar'] == CATEGORY.MARKERS[1]
        assert self.markers['bazz'] == CATEGORY.MARKERS[2]
    
    def test_sizes(self):
        s_min, s_max = CATEGORY.MARKERSIZE_LIMITS
        assert self.sizes_l[1.5] == s_min
        assert self.sizes_l[3.5] == s_max
        assert self.sizes_s[1.5] == s_min**2
        assert self.sizes_s[3.5] == s_max**2

        sizes = self.sizes_l([1.5, 3.5])
        assert np.array_equal(sizes, CATEGORY.MARKERSIZE_LIMITS)

        values = np.linspace(
            self.sizes_l._min, self.sizes_l._max, CATEGORY.N_SIZE_BINS)
        sizes = self.sizes_l(values)
        assert np.array_equal(sizes, self.sizes_l.categories)

        sizes = self.sizes_l([1.5, 3.5])
        assert np.array_equal(sizes, CATEGORY.MARKERSIZE_LIMITS)

        values = np.linspace(
            self.sizes_l._min, self.sizes_l._max, CATEGORY.N_SIZE_BINS)
        sizes = self.sizes_s(values)
        assert np.array_equal(sizes, np.square(self.sizes_s.categories))


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
            
            if i in [0, 1]:
                assert self.axs.row_idx == 0
            else:
                assert self.axs.row_idx == 1
            
            if i in [0, 2]:
                assert self.axs.col_idx == 0
            else:
                assert self.axs.col_idx == 1

    def test_label_facets(self):...


class TestCharts:

    def test_multiple_variate_plot(self):
        chart = MultipleVariateChart(
            source = affairs.data,
            target = 'affairs',
            feature = 'yrs_married', 
            hue = 'rate_marriage',
            size = 'age',
            shape = 'educ',
            col = 'religious',
            row = 'children'
            )
        chart.plot(Scatter)
        chart.label(
            fig_title = 'Multiple Variate Chart',
            sub_title = 'MTCars R Dataset',
            xlabel = 'Needed time for 1/4 mile',
            ylabel = 'miles / gallon',
            row_title = 'V-shaped engine',
            col_title = 'Amount of carburetors',
            info = 'pytest figure'
        )
        chart.save(savedir/'multivariate_chart_mtcars.png')