import sys
import pytest
import matplotlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pytest import approx
from typing import List
from pathlib import Path
from matplotlib.text import Text
from pandas.core.frame import DataFrame
from matplotlib.ticker import PercentFormatter

sys.path.append(Path(__file__).parent.resolve())

from daspi.strings import STR
from daspi.constants import PLOTTER
from daspi.plotlib.chart import JointChart
from daspi.plotlib.chart import SimpleChart
from daspi.plotlib.chart import MultipleVariateChart
from daspi.plotlib.plotter import Bar
from daspi.plotlib.plotter import Line
from daspi.plotlib.plotter import Pareto
from daspi.plotlib.plotter import Jitter
from daspi.plotlib.plotter import Scatter
from daspi.plotlib.plotter import Violine
from daspi.plotlib.plotter import MeanTest
from daspi.plotlib.plotter import Probability
from daspi.plotlib.plotter import GaussianKDE
from daspi.plotlib.plotter import VariationTest
from daspi.plotlib.plotter import LinearRegression
from daspi.plotlib.plotter import StandardErrorMean


matplotlib.use("Agg")

savedir = Path(__file__).parent/'charts'
savedir.mkdir(parents=True, exist_ok=True)
df_affairs: DataFrame = sm.datasets.fair.load_pandas().data

df_travel: DataFrame = sm.datasets.modechoice.load_pandas().data
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

source = Path(__file__).parent/'data'
KW_READ = dict(sep=';', index_col=0, skiprows=1)
df_dist10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', nrows=10, **KW_READ)
df_dist25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', nrows=25, **KW_READ)

def get_texts(chart: SimpleChart) -> List[Text]:
    return sorted(chart.figure.texts, key=lambda t: t._y, reverse=True)


class TestSimpleChart:
    
    fig_title: str = 'SimpleChart'
    _sub_title: str = 'Traveling Mode Dataset'
    target_label: str = 'In vehicle cost ($)'
    feature_label: str = 'Individual'
    target: str = 'gc'
    feature: str = 'hinc'
    kind: str = ''
    info_msg: str = 'Pytest figure, additional info message'
    cat1 = 'mode'
    cat2 ='choice'
    size = 'invt'

    @property
    def sub_title(self) -> str:
        return f'{self._sub_title}: {self.kind}'

    def test_line_plot(self) -> None:
        base = f'{self.fig_title}_line'

        self.kind = 'hue'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = 'invc',
                feature = 'individual',
                hue = self.cat1
            ).plot(Line
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True, 
                info = True, 
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.cat1
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == 'invc'
        assert texts[3].get_text() == 'individual'
        assert STR.TODAY in texts[4].get_text()
        assert STR.USERNAME in texts[4].get_text()
        assert self.info_msg not in info_msg

        self.kind = 'hue_size'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart =SimpleChart(
                df_travel,
                target = 'invc',
                feature = 'individual',
                hue = self.cat1,
                size = self.size
            ).plot(Line
            ).plot(Scatter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = 'Individual',
                target_label = 'In vehicle cost ($)', 
                info = self.info_msg, 
            ).save(file_name
            ).close()
        assert file_name.is_file()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat1
        assert legend_artists[2].get_children()[0].get_text() == self.size
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == 'In vehicle cost ($)'
        assert texts[3].get_text() == 'Individual'
        assert STR.TODAY in texts[4].get_text()
        assert STR.USERNAME in texts[4].get_text()
        assert self.info_msg in info_msg
    

    def test_scatter_plot(self) -> None:
        base = f'{self.fig_title}_scatter'

        self.kind = 'simple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        assert file_name.is_file()
        assert len(texts) == 1
        assert chart.label_facets.legend_box is None
        assert texts[0].get_text() == self.sub_title


        self.kind = 'transposed'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart =SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
            ).plot(Scatter, target_on_y=False
            ).label(
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        assert file_name.is_file()
        assert len(texts) == 3
        assert chart.label_facets.legend_box is None
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.feature
        assert texts[2].get_text() == self.target


        self.kind = 'hue'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
                hue = self.cat1
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 4
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.cat1
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target_label
        assert texts[2].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg


        self.kind = 'shape'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
                shape = self.cat2
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 4
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.cat2
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target_label
        assert texts[2].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg


        self.kind = 'size'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
                size = self.size
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 4
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.size
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target_label
        assert texts[2].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg


        self.kind = 'hue-size'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
                hue = self.cat1,
                size = self.size
            ).plot(Scatter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat1
        assert legend_artists[2].get_children()[0].get_text() == self.size
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg

        self.kind = 'hue-shape'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
                hue = self.cat1,
                shape = self.cat2
            ).plot(Scatter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat1
        assert legend_artists[2].get_children()[0].get_text() == self.cat2
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg

        self.kind = 'size-shape'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
                size = self.size,
                shape = self.cat2
            ).plot(Scatter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat2
        assert legend_artists[2].get_children()[0].get_text() == self.size
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg


        self.kind = 'full'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.feature,
                hue = self.cat1,
                size = self.size,
                shape = self.cat2
            ).plot(Scatter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 6
        assert legend_artists[0].get_children()[0].get_text() == self.cat1
        assert legend_artists[2].get_children()[0].get_text() == self.cat2
        assert legend_artists[4].get_children()[0].get_text() == self.size
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
    
    def test_pareto_plot(self) -> None:
        base = f'{self.fig_title}_pareto'

        self.kind = 'simple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.cat1,
            ).plot(
                Pareto, method='sum'  
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert chart.label_facets.legend_box is None
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.cat1
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

        self.kind = 'transposed'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                feature = self.cat1,
                target_on_y = False
            ).plot(
                Pareto, method='sum'  
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert chart.label_facets.legend_box is None
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.cat1
        assert texts[3].get_text() == self.target_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg


    def test_kde_plot(self) -> None:
        base = f'{self.fig_title}_KDE'

        self.kind = 'simple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                target_on_y = False
            ).plot(
                GaussianKDE, show_density_axis=True
            ).label(
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        assert file_name.is_file()
        assert len(texts) == 2 # feature label should not appear
        assert chart.label_facets.legend_box is None
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target

        self.kind = 'multiple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                df_travel,
                target = self.target,
                hue = self.cat1,
                target_on_y = True
            ).plot(
                GaussianKDE, show_density_axis=False
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True, 
                info = True
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 4 # feature label should not appear
        assert legend_artists[0].get_children()[0].get_text() == self.cat1
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg
    
    def test_jitter_plot(self) -> None:
        base = f'{self.fig_title}_jitter'

        self.kind = 'simple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = self.target,
                feature = self.cat1, 
                categorical_features = True,
            ).plot(Jitter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'multiple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = self.target,
                feature = self.cat1, 
                hue = self.cat2,
                dodge = True
            ).plot(Jitter, target_on_y=False
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg


    def test_violine_plot(self) -> None:
        base = f'{self.fig_title}_violine'

        self.kind = 'mono'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = self.target,
            ).plot(Violine
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 4 # feature label should not appear
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

        self.kind = 'simple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = self.target,
                feature = self.cat1, 
            ).plot(Violine
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'multiple'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = self.target,
                feature = self.cat1, 
                hue = self.cat2,
                dodge = True
            ).plot(Violine, target_on_y=False
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

    def test_errorbar_plots(self) -> None:
        base = f'{self.fig_title}_errorbar'

        self.kind = 'sem'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = self.target,
                feature = self.cat1, 
                hue = self.cat2,
                dodge = True,
                target_on_y = True
            ).plot(
                StandardErrorMean
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'mean-test'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode',
                categorical_features = True,
                target_on_y = False
            ).plot(
                MeanTest
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'var-test'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode',
                categorical_features = True,
                target_on_y = False
            ).plot(
                VariationTest,
                kind = 'variance'
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'std-test'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = SimpleChart(
                source = df_travel,
                target = 'gc',
                feature = 'mode',
                categorical_features = True,
                target_on_y = False,
            ).plot(
                VariationTest,
                kind = 'stdev',
                show_center = False
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg


class TestJointChart:

    fig_title: str = 'JointChart'
    _sub_title: str = 'Traveling Mode Dataset'
    target_label: str = 'In vehicle cost ($)'
    feature_label: str = 'Individual'
    target: str = 'gc'
    feature: str = 'hinc'
    kind: str = ''
    info_msg: str = 'Pytest figure, additional info message'
    cat1 = 'mode'
    cat2 ='choice'
    size = 'invt'

    @property
    def sub_title(self) -> str:
        return f'{self._sub_title}: {self.kind}'

    def test_kde_plots(self) -> None:
        base = f'{self.fig_title}_kdes'
        
        self.kind = 'mixed-kde'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = JointChart(
                df_travel,
                target = self.target,
                feature = self.cat1,
                hue = self.cat2,
                ncols = 1,
                nrows = 2,
                sharex = True,
                dodge = (False, True),
                categorical_features = (False, True),
                target_on_y = False
            ).plot([
                (GaussianKDE, dict(show_density_axis=True)),
                (Violine, {})]
            ).label(
                feature_label = [True]*2,
                target_label = [True]*2
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_box.get_children()
        yticklabels1 = [t.get_text() for t in chart.axes[1][0].get_yticklabels()]
        xticklabels0 = [t.get_text() for t in chart.axes[0][0].get_xticklabels()]
        xticklabels1 = [t.get_text() for t in chart.axes[1][0].get_xticklabels()]
        assert file_name.is_file()
        assert len(texts) == 0 # No text added to figure only to axes
        assert legend_artists[0].get_children()[0].get_text() == self.cat2
        assert yticklabels1 == sorted(df_travel[self.cat1].unique())
        assert bool(xticklabels0) == False
        assert bool(xticklabels1) == True
        assert chart.axes[0][0].get_xlabel() == ''
        assert chart.axes[1][0].get_xlabel() == self.target
        assert chart.axes[1][0].get_ylabel() == self.cat1
    

    def test_probabilities(self) -> None:
        base = f'{self.fig_title}_probability'
        
        self.kind = 'norm-prob'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = JointChart(
                df_dist25,
                target = 'rayleigh',
                feature = '',
                nrows = 2,
                ncols = 2,
                target_on_y = False
            ).plot([
                (Probability, {'kind': 'qq'}),
                (Probability, {'kind': 'pp'}),
                (Probability, {'kind': 'sq'}),
                (Probability, {'kind': 'sp'})]
            ).label(
                fig_title = self.fig_title,
                sub_title = 'QQ, PP, samples-Q and samples-P',
                target_label = (
                    'norm quantiles', 'norm percentiles',
                    'norm samples', 'norm samples'),
                feature_label = (
                    'theoretical quantiles', 'theoretical percentiles',
                    'theoretical quantiles', 'theoretical percentiles')
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        assert file_name.is_file()
        assert len(texts) == 2
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == 'QQ, PP, samples-Q and samples-P'
        assert chart.axes[0][0].get_ylabel() == 'theoretical quantiles'
        assert chart.axes[0][1].get_ylabel() == 'theoretical percentiles'
        assert chart.axes[1][0].get_ylabel() == 'theoretical quantiles'
        assert chart.axes[1][1].get_ylabel() == 'theoretical percentiles'
        assert chart.axes[0][0].get_xlabel() == 'norm quantiles'
        assert chart.axes[0][1].get_xlabel() == 'norm percentiles'
        assert chart.axes[1][0].get_xlabel() == 'norm samples'
        assert chart.axes[1][1].get_xlabel() == 'norm samples'
        for l in [t.get_text() for t in chart.axes[0][0].get_yticklabels()]:
            assert '%' not in l
        for l in [t.get_text() for t in chart.axes[0][1].get_yticklabels()]:
            assert '%' in l
        for l in [t.get_text() for t in chart.axes[1][0].get_yticklabels()]:
            assert '%' not in l
        for l in [t.get_text() for t in chart.axes[1][1].get_yticklabels()]:
            assert '%' in l
        for l in [t.get_text() for t in chart.axes[0][0].get_xticklabels()]:
            assert '%' not in l
        for l in [t.get_text() for t in chart.axes[0][1].get_xticklabels()]:
            assert '%' in l
        for l in [t.get_text() for t in chart.axes[1][0].get_xticklabels()]:
            assert '%' not in l
        for l in [t.get_text() for t in chart.axes[1][1].get_xticklabels()]:
            assert '%' not in l

        self.kind = 'dists-prob'
        file_name = savedir/f'{base}_{self.kind}.png'
        target = df_dist25.columns.to_list()[1:]
        target_labels = tuple(f'{d} quantiles' for d in target)
        feature_label = 'theoretical quantiles'
        chart = JointChart(
                df_dist25,
                target = target,
                feature = '',
                nrows = 3,
                ncols = 3
            ).plot([
                (Probability, dict(dist=d, kind='qq')) for d in target]
            ).label(
                fig_title = self.fig_title,
                sub_title = 'QQ for different distributions',
                target_label = target_labels,
                feature_label = feature_label,
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 4
        for ax, ylabel in zip(chart.axes.flat, target_labels):
            assert ax.get_xlabel() == ''
            assert ax.get_ylabel() == ylabel
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == 'QQ for different distributions'
        assert texts[2].get_text() == feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

    def test_regression_joint(self) -> None:
        base = f'{self.fig_title}_probability'

        self.kind = 'kde'
        file_name = savedir/f'{base}_{self.kind}.png'
        chart = JointChart(
                source = df_travel,
                target = ('invc', '', 'invt', 'invt'),
                feature = ('', '', 'invc', ''),
                target_on_y = [False, False, True, True],
                hue = self.cat1,
                nrows = 2,
                ncols = 2,
                sharex = 'col',
                sharey = 'row',
                width_ratios = [5, 1],
                height_ratios = [1, 5],
                stretch_figsize = False
        ).plot([
            (GaussianKDE, {'show_density_axis': False}),
            (None, {}),
            (LinearRegression, dict(show_center=True, show_fit_ci=True, show_pred_ci=True)),
            (GaussianKDE, {'show_density_axis': False})]
        ).label(
        ).save(file_name
        ).close()
        assert file_name.is_file()

    def test_bar_plots(self) -> None:
        base = f'{self.fig_title}_bar'

        self.kind = 'stacking'
        file_name = savedir/f'{base}_{self.kind}.png'
        target_labels = (
            'Generalized cost measure count ($)',
            'Generalized cost measure sum ($)')
        feature_label = 'Traveler chosen mode'
        chart = JointChart(
                source = df_travel,
                target = self.target,
                feature = 'mode',
                nrows = 2,
                ncols = 1,
                hue = 'choice',
                categorical_features = True,
                target_on_y = (False, False),
                dodge = (False, True)
            ).plot([
                (Bar, dict(method='count')),
                (Bar, dict(method='sum'))]
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = feature_label, 
                target_label = target_labels,
                info = self.info_msg
            ).save(file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert file_name.is_file()
        assert len(texts) == 4
        for ax, xlabel in zip(chart.axes.flat, target_labels):
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ''
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg


class TestMultipleVariateChart:

    fig_title: str = 'MultipleVariateChart'
    _sub_title: str = 'Traveling Mode Dataset'

    @property
    def sub_title(self) -> str:
        return f'{self._sub_title}: {self.kind}'

    def test_full(self) -> None:
        base = f'{self.fig_title}'

        self.kind = 'full'
        file_name = savedir/f'{base}_{self.kind}.png'
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
                feature_label = 'Years of marriage',
                target_label = 'Amount of affairs',
                row_title = 'Amount of children',
                col_title = 'How religious',
                info = 'pytest figure')
        chart.save(file_name).close()
        assert file_name.is_file()

        base = f'{self.fig_title}'
        self.kind = 'full-stripes'
        file_name = savedir/f'{base}_{self.kind}.png'
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
            ).stripes(
                mean = True,
                control_limits = True,
                confidence = 0.95,
            ).label(
                fig_title = 'Multiple Variate Chart',
                sub_title = 'Affairs R Dataset',
                feature_label = 'Years of marriage',
                target_label = 'Amount of affairs',
                row_title = 'Amount of children',
                col_title = 'How religious',
                info = 'pytest figure')
        chart.save(file_name).close()
        assert file_name.is_file()
