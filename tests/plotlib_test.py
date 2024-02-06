import sys
import pytest
import matplotlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pytest import approx
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve())

from daspi._constants import CATEGORY
from daspi.plotlib.utils import Dodger
from daspi.plotlib.utils import HueLabel
from daspi.plotlib.utils import SizeLabel
from daspi.plotlib.utils import ShapeLabel
from daspi.plotlib.chart import SimpleChart
from daspi.plotlib.chart import RelationalChart
from daspi.plotlib.chart import MultipleVariateChart
from daspi.plotlib.facets import AxesFacets
from daspi.plotlib.plotter import Line
from daspi.plotlib.plotter import Ridge
from daspi.plotlib.plotter import Jitter
from daspi.plotlib.plotter import Scatter
from daspi.plotlib.plotter import Violine
from daspi.plotlib.plotter import MeanTest
from daspi.plotlib.plotter import GaussianKDE
from daspi.plotlib.plotter import VariationTest
from daspi.plotlib.plotter import StandardErrorMean

matplotlib.use("Agg")

savedir = Path(__file__).parent/'charts'
savedir.mkdir(parents=True, exist_ok=True)
df_affairs: pd.DataFrame = sm.datasets.fair.load_pandas().data

df_travel: pd.DataFrame = sm.datasets.modechoice.load_pandas().data
df_travel['mode'] = df_travel['mode'].replace(
    {1: 'air', 2: 'train', 3: 'bus', 4: 'car'})
df_travel['choice'] = df_travel['choice'].replace({0: 'no', 1: 'yes'})
"""
Number of observations: 840 Observations On 4 Modes for 210 Individuals.
Number of variables: 8
Variable name definitions::

    individual = 1 to 210
    mode =
        1 - air
        2 - train
        3 - bus
        4 - car
    choice =
        0 - no
        1 - yes
    ttme = terminal waiting time for plane, train and bus (minutes); 0
           for car.
    invc = in vehicle cost for all stages (dollars).
    invt = travel time (in-vehicle time) for all stages (minutes).
    gc = generalized cost measure:invc+(invt*value of travel time savings)
        (dollars).
    hinc = household income ($1000s).
    psize = traveling group size in mode chosen (number).
    
https://www.statsmodels.org/stable/datasets/generated/modechoice.html
"""


class TestCategoryLabel:
    colors = HueLabel(('alpha', 'beta', 'gamma', 'delta'))
    markers = ShapeLabel(('foo', 'bar', 'bazz'))
    sizes = SizeLabel(1.5, 3.5)

    def test_str(self):
        assert str(self.colors) == 'HueLabel'
        assert str(self.markers) == 'ShapeLabel'
        assert str(self.sizes) == 'SizeLabel'

    def test_errrors(self):
        with pytest.raises(KeyError) as err:
            self.colors['psi']
        assert str(err.value) == f"\"Can't get category for label 'psi', got {self.colors.labels}\""
        
        with pytest.raises(AssertionError) as err:
            n = self.colors.n_allowed
            HueLabel([i for i in range(n+1)])
        assert str(err.value) == f'HueLabel can handle {n} categories, got {n+1}'

        with pytest.raises(AssertionError) as err:
            ShapeLabel(('foo', 'foo', 'bar', 'bazz'))
        assert str(err.value) == f'One or more labels occur more than once, only unique labels are allowed'

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
        assert self.colors['alpha'] == CATEGORY.COLORS[0]
        assert self.colors['beta'] == CATEGORY.COLORS[1]
        assert self.colors['gamma'] == CATEGORY.COLORS[2]
        assert self.colors['delta'] == CATEGORY.COLORS[3]

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
        tick_labels = ('t', 'e', 's', 't')
        dodge = Dodger(('r'), tick_labels=tick_labels)
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
        tick_labels = ('t', 'e', 's', 't')
        categories = ('r', 'g', 'b', 'y')
        dodge = Dodger(categories=categories, tick_labels=tick_labels)
        old = pd.Series(categories)
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
    
    def test_line_plot(self):

        file_name = savedir/'line_chart_hue.png'
        chart =RelationalChart(
                df_travel,
                target = 'invc',
                feature = 'individual',
                hue = 'mode'
            ).plot(Line
            ).label(
                fig_title='Line diagram', sub_title='Travel Mode Choice',
                xlabel='Individual', ylabel='In vehicle cost ($)', 
                info=True, 
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'line_chart_hue-size.png'
        chart =RelationalChart(
                df_travel,
                target = 'invc',
                feature = 'individual',
                hue = 'mode',
                size = 'hinc'
            ).plot(Line
            ).plot(Scatter
            ).label(
                fig_title='Line diagram', sub_title='Travel Mode Choice',
                xlabel='Individual', ylabel='In vehicle cost ($)', 
                info=True, 
            ).save(file_name
            ).close()
        assert file_name.is_file()

    def test_scatter_plot(self):

        file_name = savedir/'scatter_chart_simple.png'
        chart =RelationalChart(
                df_travel,
                target = 'invc',
                feature = 'hinc'
            ).plot(Scatter
            ).label(
                sub_title='Simple XY scatter', xlabel=False, ylabel=False
            ).save(file_name
            ).close()
        assert file_name.is_file()
        
        file_name = savedir/'scatter_chart_simple_transposed.png'
        chart =RelationalChart(
                df_travel,
                target = 'invc',
                feature = 'hinc'
            ).plot(Scatter, target_on_y=False
            ).label(
                sub_title='Transposed XY scatter', xlabel=True, ylabel=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_hue.png'
        chart =RelationalChart(
                df_travel,
                target = 'gc',
                feature = 'hinc',
                hue = 'mode'
            ).plot(Scatter
            ).label(
                sub_title='Hue relational chart', xlabel='Houshold income ($)', 
                ylabel='Generalized cost measure ($)', info=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_shape.png'
        chart =RelationalChart(
                df_travel,
                target = 'gc',
                feature = 'hinc',
                shape = 'choice'
            ).plot(Scatter
            ).label(
                sub_title='Shape relational chart', xlabel='Houshold income ($)', 
                ylabel='Generalized cost measure ($)', info=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_size.png'
        chart =RelationalChart(
                df_travel,
                target = 'gc',
                feature = 'hinc',
                size = 'invt'
            ).plot(Scatter
            ).label(
                sub_title='Size relational chart', xlabel='Houshold income ($)', 
                ylabel='Generalized cost measure ($)', info=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_hue-size.png'
        chart =RelationalChart(
                df_travel,
                target = 'gc',
                feature = 'hinc',
                hue = 'mode',
                size = 'invt'
            ).plot(Scatter
            ).label(
                sub_title = 'Hue Size relational chart',
                xlabel = 'Houshold income ($)', 
                ylabel = 'Generalized cost measure ($)',
                info = True, 
                fig_title = 'Scatter diagram'
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_hue-shape.png'
        chart =RelationalChart(
                df_travel,
                target = 'gc',
                feature = 'hinc',
                hue = 'mode',
                shape = 'choice'
            ).plot(Scatter
            ).label(
                sub_title = 'Hue Shape relational chart',
                xlabel = 'Houshold income ($)', 
                ylabel = 'Generalized cost measure ($)',
                info = True, 
                fig_title = 'Scatter diagram'
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_size-shape.png'
        chart =RelationalChart(
                df_travel,
                target = 'gc',
                feature = 'hinc',
                size = 'invt',
                shape = 'choice'
            ).plot(Scatter
            ).label(
                sub_title = 'Size Shape relational chart', 
                xlabel = 'Houshold income ($)', 
                ylabel = 'Generalized cost measure ($)',
                info = True,
                fig_title = 'Scatter diagram'
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'scatter_chart_full.png'
        chart =RelationalChart(
                df_travel,
                target = 'gc',
                feature = 'hinc',
                hue = 'mode',
                size = 'invt',
                shape = 'choice'
            ).plot(Scatter
            ).label(
                sub_title = 'Size Shape XY scatter', 
                xlabel = 'Houshold income ($)', 
                ylabel = 'Generalized cost measure ($)',
                info = 'pytest figure', 
                fig_title = 'Scatter diagram'
            ).save(file_name
            ).close()
        assert file_name.is_file()

    def test_kde_plot(self):

        

        file_name = savedir/'kde_chart_simple.png'
        chart = SimpleChart(
                df_travel,
                target = 'gc',
            ).plot(
                GaussianKDE, target_on_y=False
            ).label(
                sub_title='Simple KDE', xlabel=True, ylabel=True
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'kde_chart_multiple.png'
        chart = SimpleChart(
                df_travel,
                target = 'gc',
                hue = 'mode'
            ).plot(
                GaussianKDE, target_on_y=True, show_density_axis=False
            ).label(
                sub_title='multiple by hue', xlabel=True, ylabel=True, 
                info=True, fig_title='Kernel Density Estimation'
            ).save(file_name
            ).close()
        assert file_name.is_file()

        file_name = savedir/'kde_chart_ridge.png'
        chart = SimpleChart(
                df_travel,
                target = 'gc',
                feature = 'mode',
                categorical_features = True,
            ).plot(
                Ridge, target_on_y=False,
            ).label(
                sub_title='multiple by hue', xlabel=True, ylabel=True, 
                info=True, fig_title='Kernel Density Estimation'
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
                row = 'children',
            ).plot(Scatter
            ).label(
                fig_title = 'Multiple Variate Chart',
                sub_title = 'Affairs R Dataset',
                xlabel = 'Years of marriage',
                ylabel = 'Amount of affairs',
                row_title = 'Amount of children',
                col_title = 'How religious',
                info = 'pytest figure')
        chart.save(file_name).close()
        assert file_name.is_file()

    def test_jitter_plot(self):
        file_name = savedir/'jitter_chart_simple.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode', 
                categorical_features = True,
            ).plot(Jitter
            ).label(
                fig_title = 'Violine Chart',
                sub_title = 'Simple test plot',
                xlabel = 'Traveler chosen mode', 
                ylabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()
        
        file_name = savedir/'jitter_chart_multiple.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode', 
                hue = 'choice',
                dodge = True
            ).plot(Jitter, target_on_y=False
            ).label(
                fig_title = 'Jitter Chart',
                sub_title = 'Simple test plot',
                xlabel = 'Traveler chosen mode', 
                ylabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()


    def test_violine_plot(self):
        file_name = savedir/'violine_chart_simple.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode', 
            ).plot(Violine
            ).label(
                fig_title = 'Violine Chart',
                sub_title = 'Simple test plot',
                xlabel = 'Traveler chosen mode', 
                ylabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()
        
        file_name = savedir/'violine_chart_multiple.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode', 
                hue = 'choice',
                dodge = True
            ).plot(Violine, target_on_y=False
            ).label(
                fig_title = 'Violine Chart',
                sub_title = 'Simple test plot',
                xlabel = 'Traveler chosen mode', 
                ylabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()

    def test_errorbar_plots(self):
        file_name = savedir/'errbar_sem_chart.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode',
                hue = 'choice',
                dodge = True,
            ).plot(
                StandardErrorMean,
                target_on_y = True
            ).label(
                fig_title = 'Errorbar Chart',
                sub_title = 'Standard error mean',
                xlabel = 'Traveler chosen mode', 
                ylabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()
        
        file_name = savedir/'errbar_mean_test_chart.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode',
                categorical_features = True,
            ).plot(
                MeanTest,
                target_on_y = False
            ).label(
                fig_title = 'Mean Test Plot',
                sub_title = 'Test mean',
                ylabel = 'Traveler chosen mode', 
                xlabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()
        
        file_name = savedir/'errbar_var_test_chart.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode',
                categorical_features = True,
            ).plot(
                VariationTest,
                target_on_y = False,
                kind = 'variance'
            ).label(
                fig_title = 'Variation Test Plot',
                sub_title = 'Test Variance',
                ylabel = 'Traveler chosen mode', 
                xlabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()
        
        file_name = savedir/'errbar_std_test_chart.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode',
                categorical_features = True,
            ).plot(
                VariationTest,
                target_on_y = False,
                kind = 'stdev',
                show_points = False
            ).label(
                fig_title = 'Variation Test Plot',
                sub_title = 'Test Standardeviation',
                ylabel = 'Traveler chosen mode', 
                xlabel = 'Generalized cost measure ($)',
                info = 'pytest figure'
            ).save(file_name
            ).close()
