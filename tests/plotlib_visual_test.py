import sys
import pytest
import matplotlib

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from matplotlib.text import Text
from pandas.core.frame import DataFrame

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi import load_dataset
from daspi import STR
from daspi import COLOR
from daspi import Bar
from daspi import Line
from daspi import Pareto
from daspi import Jitter
from daspi import Scatter
from daspi import Violine
from daspi import MeanTest
from daspi import JointChart
from daspi import SingleChart
from daspi import HideSubplot
from daspi import BlandAltman
from daspi import Probability
from daspi import GaussianKDE
from daspi import VariationTest
from daspi import ResiduesCharts
from daspi import LinearRegression
from daspi import StandardErrorMean
from daspi import MultipleVariateChart
from daspi import PairComparisonCharts
from daspi import ParameterRelevanceCharts
from daspi import BivariateUnivariateCharts


matplotlib.use("Agg")

savedir = Path(__file__).parent/'charts'
savedir.mkdir(parents=True, exist_ok=True)
df_aspirin = load_dataset('aspirin-dissolution')
"""
Number of observations: 48
Number of features: 8 is composed of 5 factor levels and 3 covariates
Target name: time
    sum of preparation and dissolution
    [time] = seconds
Feature name difinitions:
    employee = 3 levels
        - A
        - B
        - C
    stirrer = 2 levels
        - wooden
        - metal
    brand = 2 levels
        - ASS
        - Godamed
    catalyst = 2 levels
        - False
        - True
    water = 2 levels
        - carbonated
        - tap
    temperature = continuous (covariate)
        [temperature] = °C
    preparation = continuous (covariate)
        [preparation] = s
    dissolution = continuous (covariate)
        [dissolution] = s
"""
df_affairs: DataFrame = sm.datasets.fair.load_pandas().data # type: ignore

df_travel: DataFrame = sm.datasets.modechoice.load_pandas().data # type: ignore
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
KW_READ: Dict[str, Any] = dict(sep=';', index_col=0, skiprows=1)
df_dist10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', nrows=10, **KW_READ)
df_dist25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', nrows=25, **KW_READ)

def get_texts(chart: SingleChart | JointChart | MultipleVariateChart) -> List[Text]:
    return sorted(chart.figure.texts, key=lambda t: t._y, reverse=True) # type: ignore


class TestSingleChart:
    
    fig_title: str = 'SingleChart'
    _sub_title: str = 'Aspirin Dissolution Dataset'
    target_label: str = 'Dissolution time (s)'
    feature_label: str = 'Water temperature (°C)'
    target: str = 'dissolution'
    feature: str = 'temperature'
    kind: str = ''
    info_msg: str = 'Pytest figure, additional info message'
    cat1 = 'employee'
    cat2 ='brand'
    size = 'preparation'

    @property
    def sub_title(self) -> str:
        return f'{self._sub_title}: {self.kind}'
    
    @property
    def file_name(self) -> Path:
        return savedir/f'{self.base}_{self.kind}.png'

    def test_line_plot(self) -> None:
        self.base = f'{self.fig_title}_line'

        self.kind = 'hue'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
                hue = self.cat1
            ).plot(Line
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True, 
                info = True, 
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.cat1 # type: ignore
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target
        assert texts[3].get_text() == self.feature
        assert STR.TODAY in texts[4].get_text()
        assert STR.USERNAME in texts[4].get_text()
        assert self.info_msg not in info_msg

        self.kind = 'hue_size'
        chart =SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
                hue = self.cat1,
                size = self.size
            ).plot(Line
            ).plot(Scatter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = self.info_msg, 
            ).save(self.file_name
            ).close()
        assert self.file_name.is_file()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat1 # type: ignore
        assert legend_artists[2].get_children()[0].get_text() == self.size # type: ignore
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in texts[4].get_text()
        assert STR.USERNAME in texts[4].get_text()
        assert self.info_msg in info_msg
    

    def test_scatter_plot(self) -> None:
        self.base = f'{self.fig_title}_scatter'

        self.kind = 'simple'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        assert self.file_name.is_file()
        assert len(texts) == 1
        assert chart.label_facets.legend is None
        assert texts[0].get_text() == self.sub_title

        self.kind = 'simple-stripes'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature
            ).plot(Scatter
            ).stripes(
                mean=True,
                median=True,
                control_limits=True,
                spec_limits=(8, 35),
            ).label(
                sub_title = self.sub_title
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        assert self.file_name.is_file()
        assert len(texts) == 1
        assert chart.label_facets.legend is not None
        assert texts[0].get_text() == self.sub_title

        self.kind = 'simple-stripes-65ci'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature
            ).plot(Scatter
            ).stripes(
                mean=True,
                median=True,
                control_limits=True,
                confidence=0.65,
            ).label(
                sub_title = self.sub_title
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        assert self.file_name.is_file()
        assert len(texts) == 1
        assert chart.label_facets.legend is not None
        assert texts[0].get_text() == self.sub_title

        self.kind = 'simple-stripes-99ci'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature
            ).plot(Scatter
            ).stripes(
                mean=True,
                median=True,
                control_limits=True,
                confidence=0.99,
            ).label(
                sub_title = self.sub_title
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        assert self.file_name.is_file()
        assert len(texts) == 1
        assert chart.label_facets.legend is not None
        assert texts[0].get_text() == self.sub_title

        self.kind = 'transposed'
        chart =SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
            ).plot(Scatter, target_on_y=False
            ).label(
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        assert self.file_name.is_file()
        assert len(texts) == 3
        assert chart.label_facets.legend is None
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.feature
        assert texts[2].get_text() == self.target


        self.kind = 'hue'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
                hue = self.cat1
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 4
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.cat1 # type: ignore
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target_label
        assert texts[2].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg


        self.kind = 'shape'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
                shape = self.cat2
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 4
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.cat2 # type: ignore
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target_label
        assert texts[2].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg


        self.kind = 'size'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
                size = self.size
            ).plot(Scatter
            ).label(
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label, 
                info = True
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 4
        assert len(legend_artists) == 2
        assert legend_artists[0].get_children()[0].get_text() == self.size # type: ignore
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target_label
        assert texts[2].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg


        self.kind = 'hue-size'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat1 # type: ignore
        assert legend_artists[2].get_children()[0].get_text() == self.size # type: ignore
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg

        self.kind = 'hue-shape'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat1 # type: ignore
        assert legend_artists[2].get_children()[0].get_text() == self.cat2 # type: ignore
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg

        self.kind = 'size-shape'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 4
        assert legend_artists[0].get_children()[0].get_text() == self.cat2 # type: ignore
        assert legend_artists[2].get_children()[0].get_text() == self.size # type: ignore
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg

        self.kind = 'full'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert len(legend_artists) == 6
        assert legend_artists[0].get_children()[0].get_text() == self.cat1 # type: ignore
        assert legend_artists[2].get_children()[0].get_text() == self.cat2 # type: ignore
        assert legend_artists[4].get_children()[0].get_text() == self.size # type: ignore
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
    
    def test_pareto_plot(self) -> None:
        self.base = f'{self.fig_title}_pareto'
        with pytest.raises(AssertionError) as err:
            chart = SingleChart(
                    source = df_aspirin,
                    target = self.target,
                    feature = self.cat1,
                    categorical_feature = True
                ).plot(
                    Pareto, method='sum')
        assert 'categorical_feature' in str(err.value)
        
        self.kind = 'simple'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert chart.label_facets.legend is None
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target_label
        assert texts[3].get_text() == self.cat1
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

        self.kind = 'transposed'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert chart.label_facets.legend is None
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.cat1
        assert texts[3].get_text() == self.target_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
    def test_kde_plot(self) -> None:
        self.base = f'{self.fig_title}_KDE'

        self.kind = 'simple'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                target_on_y = False
            ).plot(
                GaussianKDE, show_density_axis=True
            ).label(
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        assert self.file_name.is_file()
        assert len(texts) == 2 # feature label should not appear
        assert chart.label_facets.legend is None
        assert texts[0].get_text() == self.sub_title
        assert texts[1].get_text() == self.target

        self.kind = 'multiple'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 4 # feature label should not appear
        assert legend_artists[0].get_children()[0].get_text() == self.cat1 # type: ignore
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert texts[2].get_text() == self.target
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg not in info_msg
    
    def test_jitter_plot(self) -> None:
        self.base = f'{self.fig_title}_jitter'

        self.kind = 'simple'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1, 
                categorical_feature = True,
            ).plot(Jitter
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'multiple'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg


    def test_violine_plot(self) -> None:
        self.base = f'{self.fig_title}_violine'

        self.kind = 'mono'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
            ).plot(Violine
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 4 # feature label should not appear
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

        self.kind = 'simple'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1, 
            ).plot(Violine
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'multiple'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

    def test_errorbar_plots(self) -> None:
        self.base = f'{self.fig_title}_errorbar'

        self.kind = 'sem'
        chart = SingleChart(
                source = df_aspirin,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'mean-test'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1,
                categorical_feature = True,
                target_on_y = False
            ).plot(
                MeanTest
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,    
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'var-test'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1,
                categorical_feature = True,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'std-test'
        chart = SingleChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1,
                categorical_feature = True,
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
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
    
    def test_bland_altman_plot(self) -> None:
        self.base = f'{self.fig_title}_blandaltman'
        
        self.kind = 'simple'
        chart = SingleChart(
                source = load_dataset('shoe-sole'),
                target = 'wear',
                feature = 'status',
            ).plot(
                BlandAltman,
                identity='tester',
                feature_axis='mean',
                reverse=True)
        target_label = f'Difference {chart.plots[0].target}'
        sub_title = 'Shoe sole materials dataset'
        feature_label = 'Mean both measurements'
        chart = chart.label(
                fig_title = self.fig_title,
                sub_title = sub_title,
                feature_label = feature_label,
                target_label = target_label,    
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == sub_title
        assert texts[2].get_text() == target_label
        assert texts[3].get_text() == feature_label
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
    
    def test_categorical_float_features(self) -> None:
        self.base = f'{self.fig_title}_categorical-float-features'
        
        self.kind = 'simple'
        chart = SingleChart(
                source = df_aspirin,
                target = 'dissolution',
                feature = 'temperature',
                categorical_feature = True,
            ).plot(
                Violine,
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label,
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 5
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg


class TestJointChart:

    fig_title: str = 'JointChart'
    _sub_title: str = 'Aspirin Dissolution Dataset'
    target_label: str = 'Dissolution time (s)'
    feature_label: str = 'Water temperature (°C)'
    target: str = 'dissolution'
    feature: str = 'temperature'
    kind: str = ''
    info_msg: str = 'Pytest figure, additional info message'
    cat1 = 'employee'
    cat2 ='brand'
    size = 'preparation'

    @property
    def sub_title(self) -> str:
        return f'{self._sub_title}: {self.kind}'
    
    @property
    def file_name(self) -> Path:
        return savedir/f'{self.base}_{self.kind}.png'

    def test_raises(self):
        with pytest.raises(AssertionError) as err:
            JointChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1,
                nrows = 2,
                ncols = 2,
                target_on_y = (True, True, False, False)
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = self.target_label,
                info = self.info_msg)
        err_msg = 'Single label not allowed'
        assert err_msg in str(err.value)

    def test_kde_mean_plots(self) -> None:
        self.base = f'{self.fig_title}_kdes'
        
        self.kind = 'kde-mean'
        chart = JointChart(
                source = df_aspirin,
                target = self.target,
                feature = (self.cat1, ''),
                hue = self.cat2,
                ncols = 1,
                nrows = 2,
                sharex = True,
                dodge = (False, True),
                target_on_y = False,
                height_ratios=[5, 1],
            ).plot(GaussianKDE, show_density_axis=True
            ).plot(MeanTest
            ).label(
            #     feature_label = (True, True),
            #     target_label = (True, True)
            ).save(self.file_name
            ).close()

    def test_kde_plots(self) -> None:
        self.base = f'{self.fig_title}_kdes'
        
        self.kind = 'mixed-kde'
        chart = JointChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1,
                hue = self.cat2,
                ncols = 1,
                nrows = 2,
                sharex = True,
                dodge = (False, True),
                categorical_feature = (False, True),
                target_on_y = False
            ).plot(GaussianKDE, show_density_axis=True
            ).plot(Violine
            ).label(
                feature_label = (True, True),
                target_label = (True, True),
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        legend_artists = chart.label_facets.legend_artists
        yticklabels1 = [t.get_text() for t in chart.axes[1][0].get_yticklabels()]
        xticklabels0 = [t.get_text() for t in chart.axes[0][0].get_xticklabels()]
        xticklabels1 = [t.get_text() for t in chart.axes[1][0].get_xticklabels()]
        assert self.file_name.is_file()
        assert len(texts) == 0 # No text added to figure only to axes
        assert legend_artists[0].get_children()[0].get_text() == self.cat2 # type: ignore
        assert yticklabels1 == sorted(df_aspirin[self.cat1].unique())
        assert bool(xticklabels0) == False
        assert bool(xticklabels1) == True
        assert chart.axes[0][0].get_xlabel() == ''
        assert chart.axes[1][0].get_xlabel() == self.target
        assert chart.axes[1][0].get_ylabel() == self.cat1
    
    def test_probabilities(self) -> None:
        self.base = f'{self.fig_title}_probability'
        
        self.kind = 'norm-prob'
        chart = JointChart(
                df_dist25,
                target = 'rayleigh',
                feature = '',
                nrows = 2,
                ncols = 2,
                target_on_y = False
            ).plot(Probability, kind='qq'
            ).plot(Probability, kind='pp'
            ).plot(Probability, kind='sq'
            ).plot(Probability, kind='sp'
            ).label(
                fig_title = self.fig_title,
                sub_title = 'QQ, PP, samples-Q and samples-P',
                target_label = (
                    'norm quantiles', 'norm percentiles',
                    'norm samples', 'norm samples'),
                feature_label = (
                    'theoretical quantiles', 'theoretical percentiles',
                    'theoretical quantiles', 'theoretical percentiles')
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        assert self.file_name.is_file()
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
        target = tuple(df_dist25.columns.to_list()[1:])
        target_labels = tuple(f'{d} quantiles' for d in target)
        feature_label = 'theoretical quantiles'
        chart = JointChart(
                df_dist25,
                target = target,
                feature = '',
                nrows = 3,
                ncols = 3
            )
        for distribution in target:
            chart.plot(Probability, dist=distribution, kind='qq')
        chart.label(
                fig_title = self.fig_title,
                sub_title = 'QQ for different distributions',
                target_label = target_labels,
                feature_label = feature_label,
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
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
        self.base = f'{self.fig_title}_regression'

        self.kind = 'kde'
        target_labels = ('', '', 'In vehicle time (s)', '')
        feature_labels = ('', '', 'In vehicle cost ($)', '')
        chart = JointChart(
                source = df_travel,
                target = ('invc', '', 'invt', 'invt'),
                feature = ('', '', 'invc', ''),
                target_on_y = (False, False, True, True),
                hue = 'mode',
                nrows = 2,
                ncols = 2,
                sharex = 'col',
                sharey = 'row',
                width_ratios = [5, 1],
                height_ratios = [1, 5],
                stretch_figsize = False
        ).plot(GaussianKDE, show_density_axis=False
        ).plot(HideSubplot
        ).plot(LinearRegression, show_points=True, show_fit_ci=True
        ).plot(GaussianKDE, show_density_axis=False
        ).label(
            feature_label = feature_labels,
            target_label = target_labels,
            fig_title = self.fig_title,
            sub_title = 'Traveling Mode Dataset',
            info = self.info_msg
        ).save(self.file_name
        ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 3
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == 'Traveling Mode Dataset'
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        for ax, flabel, tlabel in zip(chart.axes.flat, feature_labels, target_labels):
            assert ax.get_xlabel() == flabel
            assert ax.get_ylabel() == tlabel

    def test_bar_plots(self) -> None:
        self.base = f'{self.fig_title}_bar'

        self.kind = 'stacking'
        target_labels = (
            'Counted observed dissolution time',
            'Dissolution time (s)')
        feature_label = 'Employee'
        chart = JointChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat2,
                nrows = 2,
                ncols = 1,
                hue = self.cat1,
                categorical_feature = True,
                target_on_y = (False, False),
                dodge = (False, True)
            ).plot(Bar, method='count'
            ).plot(Bar, method='sum'
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = feature_label, 
                target_label = target_labels,
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
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

    def test_pareto_plot(self) -> None:
        self.base = f'{self.fig_title}_pareto'
        with pytest.raises(AssertionError) as err:
            JointChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1,
                nrows = 3,
                ncols = 2,
                sharex = 'col',
            ).plot(Pareto, method='sum')
        assert 'shared with other axes' in str(err.value)

        self.kind = 'marked'
        chart = JointChart(
                source = df_aspirin,
                target = self.target,
                feature = self.cat1,
                nrows = 2,
                ncols = 2,
                target_on_y = (True, True, False, False)
            ).plot(Pareto, method='sum', highlight='C', highlighted_as_last=False
            ).plot(Pareto, method='sum', highlight='B', highlight_color=COLOR.GOOD
            ).plot(Pareto, method='sum', highlight='C', highlighted_as_last=False
            ).plot(Pareto, method='sum', highlight='B', highlight_color=COLOR.GOOD
            ).stripes(mean=True
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = (True, True, True, True),
                target_label = tuple([self.target_label]*4),
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        assert self.file_name.is_file()
        assert len(texts) == 3 # Feature and target labels are not as texts in figure here
        for i, ax in enumerate(chart.axes.flat):
            if i < 2:
                assert ax.get_xlabel() == self.cat1
                assert ax.get_ylabel() == self.target_label
            else:
                assert ax.get_xlabel() == self.target_label
                assert ax.get_ylabel() == self.cat1
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg


class TestMultipleVariateChart:

    fig_title: str = 'MultipleVariateChart'
    _sub_title: str = 'Traveling Mode Dataset'

    @property
    def sub_title(self) -> str:
        return f'{self._sub_title}: {self.kind}'
    
    @property
    def file_name(self) -> Path:
        return savedir/f'{self.base}_{self.kind}.png'

    def test_full(self) -> None:
        self.base = f'{self.fig_title}'

        self.kind = 'full'
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
        chart.save(self.file_name).close()
        assert self.file_name.is_file()

        self.base = f'{self.fig_title}'
        self.kind = 'full-stripes'
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
        chart.save(self.file_name).close()
        assert self.file_name.is_file()


class TestTemplates:

    fig_title: str = 'Templates'
    _sub_title: str = 'Aspirin Dissolution Dataset'
    target_label: str = 'Dissolution time (s)'
    feature_label: str = 'Water temperature (°C)'
    target: str = 'dissolution'
    feature: str = 'temperature'
    kind: str = ''
    base: str = ''
    info_msg: str = 'Pytest figure, additional info message'
    cat1 = 'employee'
    cat2 ='brand'
    size = 'preparation'

    @property
    def sub_title(self) -> str:
        return f'{self._sub_title}: {self.kind}'
    
    @property
    def file_name(self) -> Path:
        return savedir/f'{self.base}_{self.kind}.png'
    
    @pytest.fixture
    def df_aspirin(self) -> pd.DataFrame:
        return load_dataset('aspirin-dissolution')
    
    def test_bivariate_univariate_charts(self, df_aspirin: pd.DataFrame) -> None:
        
        self.base = f'{self.fig_title}_bivariate-univariate-charts'
        self.kind = 'simple'
        chart = BivariateUnivariateCharts(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
                hue = self.cat1
            ).plot_univariates(
                GaussianKDE
            ).plot_bivariate(
                LinearRegression
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = True,
                target_label = True,
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        xlabels = tuple(ax.get_xlabel() for ax in chart.axes.flat)
        ylabels = tuple(ax.get_ylabel() for ax in chart.axes.flat)
        assert self.file_name.is_file()
        assert len(texts) == 3
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert xlabels == ('', '', self.feature, '')
        assert ylabels == ('', '', self.target, '')
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg
        
        self.kind = 'mean-test'
        _title = '95 % confidence interval of mean'
        chart = BivariateUnivariateCharts(
                source = df_aspirin,
                target = self.target,
                feature = self.feature,
                hue = self.cat2,
                dodge_univariates=True,
                stretch_figsize=False,
            ).plot_univariates(
                MeanTest
            ).plot_bivariate(
                LinearRegression, show_fit_ci=True
            ).label(
                fig_title = self.fig_title,
                sub_title = self.sub_title,
                feature_label = self.feature_label,
                target_label = self.target_label,
                axes_titles = (_title, '', '', ''),
                info = self.info_msg
            ).save(self.file_name
            ).close()
        texts = get_texts(chart)
        info_msg = texts[-1].get_text()
        xlabels = tuple(ax.get_xlabel() for ax in chart.axes.flat)
        ylabels = tuple(ax.get_ylabel() for ax in chart.axes.flat)
        assert self.file_name.is_file()
        assert len(texts) == 3
        assert texts[0].get_text() == self.fig_title
        assert texts[1].get_text() == self.sub_title
        assert xlabels == ('', '', self.feature_label, '')
        assert ylabels == ('', '', self.target_label, '')
        assert STR.TODAY in info_msg
        assert STR.USERNAME in info_msg
        assert self.info_msg in info_msg

