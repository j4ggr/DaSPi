import pandas as pd

from io import StringIO
from pathlib import Path
from pandas.core.frame import DataFrame

dataset_info = """dataset,description,useful,ref
ancova,Teaching method with family income as covariate,ANCOVA,www.real-statistics.com
anova,Pain threshold per hair color,anova - pairwise_tukey,McClave and Dietrich 1991
anova2,Fertilizer impact on the yield of crops,anova,www.real-statistics.com
anova2_unbalanced,Diet and exercise impact,anova,http://onlinestatbook.com/2/analysis_of_variance/unequal.html
anova3,Cholesterol in different groups,anova,Pingouin
anova3_unbalanced,Cholesterol in different groups,anova,Pingouin
aspirin-dissolution,exercise improving the dissolution speed of an aspirin tablet,MultiVariateChart,Six Sigma TC GmBH
carpet_full-factorial,Full factorial DOE of a process improvement project for carpet dyeing,anova,Six Sigma TC GmbH
eeprom_full-factorial,Full factorial DOE of a process improvement project for eeproms,anova,Six Sigma TC GmbH
eeprom_partial-factorial,Partial factorial DOE of a process improvement project for eeproms,anova,Six Sigma TC GmbH
partial-factorial,Partial factorial DOE of a process improvement project,anova,Six Sigma TC GmbH
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
        Must be a valid dataset present in pingouin.datasets

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        Requested dataset.

    Examples
    --------
    Load the `Penguin <https://github.com/allisonhorst/palmerpenguins>`_
    dataset:

    >>> import daspi
    >>> df = daspi.read_dataset('penguins')
    >>> df # doctest: +SKIP
        species  island  bill_length_mm  ...  flipper_length_mm  body_mass_g     sex
    0    Adelie  Biscoe            37.8  ...              174.0       3400.0  female
    1    Adelie  Biscoe            37.7  ...              180.0       3600.0    male
    2    Adelie  Biscoe            35.9  ...              189.0       3800.0  female
    3    Adelie  Biscoe            38.2  ...              185.0       3950.0    male
    4    Adelie  Biscoe            38.8  ...              180.0       3800.0    male
    ..      ...     ...             ...  ...                ...          ...     ...
    339  Gentoo  Biscoe             NaN  ...                NaN          NaN     NaN
    340  Gentoo  Biscoe            46.8  ...              215.0       4850.0  female
    341  Gentoo  Biscoe            50.4  ...              222.0       5750.0    male
    342  Gentoo  Biscoe            45.2  ...              212.0       5200.0  female
    343  Gentoo  Biscoe            49.9  ...              213.0       5400.0    male
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
    load_dataset.__name__,
    list_dataset.__name__]
