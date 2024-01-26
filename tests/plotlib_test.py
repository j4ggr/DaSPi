import sys
import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pytest import approx
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve())

from daspi._constants import CATEGORY
from daspi.plotlib.utils import HueLabelHandler
from daspi.plotlib.utils import SizeLabelHandler
from daspi.plotlib.utils import ShapeLabelHandler
from daspi.plotlib.chart import Chart
from daspi.plotlib.chart import MultipleVariateChart
from daspi.plotlib.facets import AxesFacets
from daspi.plotlib.plotter import Scatter

savedir = Path(__file__).parent/'charts'
savedir.mkdir(parents=True, exist_ok=True)
df_affairs = sm.datasets.fair.load_pandas().data


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

    def test_scatter_plot(self):

        file_name = savedir/'scatter_chart_simple.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ'
            ).plot(Scatter
            ).label(
                sub_title='Simple XY scatter', xlabel=False, ylabel=False
            ).save(file_name
            ).close()
        assert file_name.is_file()
        
        file_name = savedir/'scatter_chart_simple_transposed.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ'
            ).plot(Scatter, target_axis='x'
            ).label(
                sub_title='Transposed XY scatter', xlabel=True, ylabel=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_hue.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ',
                hue = 'religious'
            ).plot(Scatter
            ).label(
                sub_title='Hue XY scatter', xlabel=True, ylabel=True,  
                info=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_shape.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ',
                shape = 'children'
            ).plot(Scatter
            ).label(
                sub_title='Shape XY scatter', xlabel=True, ylabel=True, 
                info=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_size.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ',
                size = 'age'
            ).plot(Scatter
            ).label(
                sub_title='Size XY scatter', xlabel=True, ylabel=True,
                info=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_hue-size.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ',
                hue = 'religious',
                size = 'age'
            ).plot(Scatter
            ).label(
                sub_title='Hue Size XY scatter', xlabel=True, ylabel=True, 
                info=True, fig_title='Scatter'
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_hue-shape.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ',
                hue = 'religious',
                shape = 'children'
            ).plot(Scatter
            ).label(
                sub_title='Hue Shape XY scatter', xlabel=True, ylabel=True, 
                info=True, fig_title='Scatter'
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_size-shape.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ',
                size = 'age',
                shape = 'children'
            ).plot(Scatter
            ).label(
                sub_title='Size Shape XY scatter', xlabel=True, ylabel=True, 
                info=True, fig_title='Scatter'
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_full.png'
        chart = Chart(
                df_affairs,
                target = 'affairs',
                feature = 'educ',
                hue = 'religious',
                size = 'age',
                shape = 'children'
            ).plot(Scatter
            ).label(
                sub_title='Size Shape XY scatter', xlabel='User x-axis label', 
                ylabel='User y-axis label', info='User info message', 
                fig_title='Scatter'
            ).save(file_name
            ).close()
        assert file_name.is_file()

    def test_multiple_variate_plot(self):
        file_name = savedir/'multivariate_chart_affairs.png'
        chart = MultipleVariateChart(
                source = df_affairs,
                target = 'affairs',
                feature = 'yrs_married', 
                hue = 'rate_marriage',
                size = 'age',
                shape = 'educ',
                col = 'religious',
                row = 'children'
            ).plot(
                Scatter
            ).label(
                fig_title = 'Multiple Variate Chart',
                sub_title = 'Affairs R Dataset',
                xlabel = 'Years of marriage',
                ylabel = 'Amount of affairs',
                row_title = 'Amount of children',
                col_title = 'How religious',
                info = 'pytest figure'
            )
        chart.axes[0][0].set(xlim=(0, 25), ylim=(0, 60))
        chart.save(file_name).close()
        assert file_name.is_file()
