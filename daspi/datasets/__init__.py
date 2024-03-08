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
blandaltman,Hypothetical data of an agreement between two methods (Method A and B). ,plot_blandaltman,https://www.biochemia-medica.com/en/journal/25/2/10.11613/BM.2015.015
chi2_independence,Patients' attributes and heart conditions,chi2_independence,https://archive.ics.uci.edu/ml/datasets/Heart+Disease
chi2_mcnemar,Responses to 2 athlete's foot treatments, chi2_mcnemar, http://www.stat.purdue.edu/~tqin/system101/method/method_mcnemar_sas.htm (adapted)
circular,Orientation tuning properties of three neurons,circular statistics,Berens 2009
cochran,Energy level across three days,Cochran,www.real-statistics.com
cronbach_alpha,Questionnaire ratings,cronbach_alpha,www.real-statistics.com
cronbach_wide_missing,Questionnaire rating (binary) in wide format and with missing values,cronbach_alpha,www.real-statistics.com
icc,Wine quality rating by 4 judges,intraclass_corr,www.real-statistics.com
mediation,Mediation analysis,linear_regression - mediation,https://data.library.virginia.edu/introduction-to-mediation-analysis/
mixed_anova,Memory scores in two groups at three time points,mixed_anova,Pingouin
mixed_anova_unbalanced,Memory scores in three groups at four time points,mixed_anova,Pingouin
multivariate,Multivariate health outcomes in drug and placebo conditions,multivariate statistics,www.real-statistics.com
pairwise_corr,Big 5 personality traits,corr - pairwise_corr,Dolan et al 2009
pairwise_tests,Scores at 3 time points per gender,pairwise_tests,Pingouin
pairwise_tests_missing,Scores at 3 time points with missing values,pairwise_tests,Pingouin
partial_corr,Scores at 4 time points,partial_corr,Pingouin
penguins,Flipper length and boody mass for different species of penguins (Adelie - Chinstrap - Gentoo),everything!,https://github.com/allisonhorst/palmerpenguins
rm_anova,Hostility towards insect,rm_anova - mixed_anova,Ryan et al 2013
rm_anova_wide,Scores at 4 time points,rm_anova,Pingouin
rm_anova2,Performance of employees at two time points and three areas,rm_anova2,www.real-statistics.com
rm_corr,Repeated measurements of pH and PaCO2,rm_corr,Bland et Altmann 1995
rm_missing,Missing values in long-format repeated measures dataframe,rm_anova - rm_anova2,Pingouin
tips,One waiter recorded information about each tip he received over a period of a few months working in one restaurant,regression, https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html
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
