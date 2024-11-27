import pandas as pd

from io import StringIO
from pathlib import Path
from pandas.core.frame import DataFrame

dataset_info = """dataset,description,useful,ref
ancova,Teaching method with family income as covariate,ANCOVA,www.real-statistics.com
anova,Pain threshold per hair color,anova,McClave and Dietrich 1991
anova2,Fertilizer impact on the yield of crops,anova,www.real-statistics.com
anova2_unbalanced,Diet and exercise impact,anova,http://onlinestatbook.com/2/analysis_of_variance/unequal.html
anova3,Cholesterol in different groups,anova,Pingouin
anova3_unbalanced,Cholesterol in different groups,anova,Pingouin
aspirin-dissolution,exercise improving the dissolution speed of an aspirin tablet,MultiVariateChart,Six Sigma TC GmBH
carpet_full-factorial,Full factorial DOE of a process improvement project for carpet dyeing,anova,Six Sigma TC GmbH
drop_card,Experiment in which an attempt is made to drop a Jass card with an outstretched arm as precisely as possible onto a center with different initial orientations of the card to the ground: (vertical and parallel). Measurements were made with a ruler,process capability,Green Belt Training Experiment at Six Sigma TC GmbH 
eeprom_full-factorial,Full factorial DOE of a process improvement project for eeproms,anova,Six Sigma TC GmbH
eeprom_partial-factorial,Partial factorial DOE of a process improvement project for eeproms,anova,Six Sigma TC GmbH
iris,This is one of the earliest datasets used in the literature on classification methods and widely used in statistics and machine learning. Here it's in a long format., BivariateUnivariate Chart,https://archive.ics.uci.edu/ml/datasets/iris
partial-factorial,Partial factorial DOE of a process improvement project,anova,Six Sigma TC GmbH
salt_sticks,Experiment in which a salt stick is broken into thirds by hand as accurately as possible. The measurements were taken with a ruler,process capability,Green Belt Training Experiment at Six Sigma TC GmbH
shoe-sole,Hypothetical data of an agreement between two shoe sole materials (old and new).,BlandAltman Plotter,Bland J. M. & Altman D. (1986)
tips,One waiter recorded information about each tip he received over a period of a few months working in one restaurant,anova, https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html
"""
df_info = pd.read_csv(StringIO(dataset_info), sep=',')

SOURCE_DIR = Path(__file__).parent
DATASET_NAMES = tuple(df_info['dataset'])

def load_dataset(dataset_name: str) -> DataFrame:
    """Load an example datasets.

    Parameters
    ----------
    dataset_name : string
        Name of dataset to load (without extension).
        Must be a valid dataset present in `daspi.datasets`

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        Requested dataset.

    Examples
    --------
    Load the `Iris <https://archive.ics.uci.edu/dataset/53/iris>`_
    dataset:

    >>> import daspi
    >>> df = daspi.load_dataset('iris')
    >>> df
        species   leaf  width  length
    0       setosa  sepal    3.5     5.1
    1       setosa  sepal    3.0     4.9
    2       setosa  sepal    3.2     4.7
    3       setosa  sepal    3.1     4.6
    4       setosa  sepal    3.6     5.0
    ..         ...    ...    ...     ...
    295  virginica  petal    2.3     5.2
    296  virginica  petal    1.9     5.0
    297  virginica  petal    2.0     5.2
    298  virginica  petal    2.3     5.4
    299  virginica  petal    1.8     5.1

    [300 rows x 4 columns]
    """
    assert dataset_name in DATASET_NAMES, (
        f'Dataset does not exist. Valid datasets names are {DATASET_NAMES}')
    return pd.read_csv(SOURCE_DIR/f'{dataset_name}.csv', sep=',')

def list_dataset():
    """List available example datasets.

    Returns
    -------
    datasets : :py:class:`pandas.DataFrame`
        A dataframe with the name, description and reference of all the
        datasets included in DaSPi.

    Examples
    --------

    >>> import daspi as pg
    >>> all_datasets = pg.list_dataset()
    >>> all_datasets.index.tolist()
    ['ancova',
     'anova',
     'anova2',
     'anova2_unbalanced',
     'anova3',
     'anova3_unbalanced',
     'blandaltman',
     'chi2_independence',
     'chi2_mcnemar',
     'circular',
     'cochran',
     'cronbach_alpha',
     'cronbach_wide_missing',
     'icc',
     'mediation',
     'mixed_anova',
     'mixed_anova_unbalanced',
     'multivariate',
     'pairwise_corr',
     'pairwise_tests',
     'pairwise_tests_missing',
     'partial_corr',
     'penguins',
     'rm_anova',
     'rm_anova_wide',
     'rm_anova2',
     'rm_corr',
     'rm_missing',
     'tips']
    """
    return df_info.set_index("dataset")

__all__ = [
    'load_dataset',
    'list_dataset']
