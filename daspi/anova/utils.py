import patsy
import itertools

import pandas as pd 

from typing import List
from typing import Dict
from typing import Tuple
from typing import DefaultDict
from collections import defaultdict
from pandas.core.frame import DataFrame
from patsy.design_info import DesignInfo
from statsmodels.regression.linear_model import RegressionResults

from ..constants import ANOVA


def is_encoded_categorical(feature_name: str) -> bool:
    """True if given feature is a patsy encoded categorical feature."""
    return bool(ANOVA.CAT_VALUE.findall(feature_name))

def clean_categoricals(feature_name: str) -> str:
    """Clean feature name of patsy encoded categorical features. Such 
    names are structured as follows '<term name>[T.<value>]'.
    Cleaned name looks like '<term name>_<value>'."""
    if ANOVA.SEP in feature_name:
        split = feature_name.split(ANOVA.SEP)
        return ANOVA.SEP.join([clean_categoricals(f) for f in split])
    else:
        name = ANOVA.CAT_ORIG.findall(feature_name)
        value = ANOVA.CAT_VALUE.findall(feature_name)
        return f'{name[0]}_{value[0]}' if name and value else feature_name

def get_term_name(feature_name):
    """Get original term name of a patsy encoded categorical feature.
    Such names are structured as follows '<term name>[T.<value>]'."""
    match = ANOVA.CAT_ORIG.findall(feature_name)
    return feature_name if not match else match[0]

def decode_cat_main(
        feature_name: str, code: int, di: DesignInfo) -> str|None:
    """Decode 0 or 1 value of encoded categorical main feature. 
    The design info coming from patsy is used to get original values."""
    matches = ANOVA.CAT_VALUE.findall(feature_name)
    if code == 1:
        return ' & '.join(matches)
    else:
        term_name = get_term_name(feature_name)
        for factor, info in di.factor_infos.items():
            if term_name != factor.code: continue
            return ' | '.join([c for c in info.categories if c not in matches])

def encoded_dmatrices(
        data: DataFrame, formula: str
        ) -> Tuple[DataFrame, DataFrame, Dict[str, tuple]]:
    """encode all levels of main factors such that the lowest level is 
    -1 and the highest level is 1. The interactions are the product of 
    the main factors. Categorical values are one-hot encoded. 
    
    Parameters
    ----------
    data : pandas DataFrame
        Data with real factor levels
    formula : str
        The formula specifying the model used by patsy.
    
    Returns
    -------
    y : pandas DataFrame
        target values
    X_code : pandas DataFrame
        Encoded feature values
    mapper : dict
        information about how the levels are encoded
        - key: str = feature name of main level
        - value: dict = key: originals, value: codes
    """
    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    design_info = X.design_info
    features = list(design_info.column_name_indexes.keys())
    X_code = pd.DataFrame(0, index=X.index, columns=features)
    X_code.design_info = design_info
    mapper = {}
    columns = {}
    for f in features:
        columns[f] = clean_categoricals(f)
        if f == ANOVA.INTERCEPT:
            x_encoded = X[f]
        elif is_encoded_categorical(f) and ANOVA.SEP not in f:
            x_encoded = X[f]
            mapper[columns[f]] = {
                decode_cat_main(f, i, design_info): i for i in (0, 1)}
        elif ANOVA.SEP not in f:
            uniques = sorted(X[f].unique())
            codes = np.linspace(-1, 1, len(uniques))
            mapper[f] = {u: c for u, c in zip(uniques, codes)}
            x_encoded = X[f].replace(mapper[f])
        else:
            split = f.split(ANOVA.SEP)
            x_encoded = X_code[split[0]]
            for s in split[1:]:
                x_encoded = x_encoded*X_code[s]
        X_code[f] = x_encoded
    X_code = X_code.rename(columns=columns)
    return y, X_code, mapper

def hierarchical(features: List[str]) -> List[str]:
    """Get all features such that all lower interactions and main 
    effects are present with the same features that appear in the higher 
    interactions

    Parameters
    ----------
    features : list of str
        Columns of exogene variables
    
    Returns
    -------
    features_h : list of str
        features for hierarchical model"""

    features_h = set(features)
    for feature in features:
        split = feature.split(ANOVA.SEP)
        n_splits = len(split)
        for s in split:
            features_h.add(s)
        if n_splits <= ANOVA.SMALLEST_INTERACTION: continue
        for i in range(ANOVA.SMALLEST_INTERACTION, n_splits):
            for combo in map(ANOVA.SEP.join, itertools.combinations(split, i)):
                features_h.add(combo)
    return sorted(list(features_h), key=lambda x: x.count(ANOVA.SEP))

def is_hierarchical(features: List[str]) -> bool:
    """Check if given features could be used for a hierarchical model"""
    features_h = hierarchical(features)
    return all([f in features for f in features_h])

def goodness_of_fit_metrics(
        model: RegressionResults, metrics: DefaultDict[str, list]=None, 
        **kwargs) -> DefaultDict[str, list]:
    """Get different goodness-of-fit metric as dict
    
    Keys:
    - 'aic' = Akaike's information criteria
    - 'rsquared' = R-squared of the model
    - 'rsquared_adj' = adjusted R-squared
    - 'p_least' = The least two-tailed p value for the t-stats of the 
    params.
    - 'hierarchical' = boolean value if model is hierarchical
    
    Parameters
    ----------
    model : statsmodels RegressionResults
        fitted OLS model
    metrics : dict of float, optional
        calculated metrics coming from last call, if none is given a 
        new defaultdict with lists is created.
    kwargs : dict
        used for additional key value pairs e.g. ols formula, least 
        param and p-value threshold (alpha) see api function 
        "recursive_feature_elimination"

    Returns
    -------
    metrics : dict
        calculated goodness-of-fit metrics
    """
    if not metrics: metrics = defaultdict(list)
    for k, v in kwargs.items():
        metrics[k].append(v)
    for k in ['aic', 'rsquared', 'rsquared_adj']:
        metrics[k].append(getattr(model, k))
    metrics['p_least'].append(model.pvalues.max())
    metrics['hierarchical'].append(is_hierarchical(list(model.params.index)))
    return metrics


__all__ = [
    is_encoded_categorical.__name__,
    clean_categoricals.__name__,
    get_term_name.__name__,
    decode_cat_main.__name__,
    encoded_dmatrices.__name__,
    hierarchical.__name__,
    is_hierarchical.__name__,
    goodness_of_fit_metrics.__name__,
]
