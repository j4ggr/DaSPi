from scipy import stats
from scipy.stats import rv_continuous


def convert_to_continuous(dist: str | rv_continuous) -> rv_continuous:
    """
    Converts a distribution to a rv_continuous object if the input is a
    string representing a distribution.
    
    Parameters
    ----------
    dist : str or rv_continuous
        The distribution to convert. Can be either a string representing
        a distribution or a rv_continuous object.
    
    Returns
    -------
    rv_continuous
        The converted rv_continuous object if the input is a
        string representing a distribution, otherwise returns the input
        distribution directly.
    """
    if isinstance(dist, str):
        return getattr(stats, dist)
    else:
        return dist

__all__ = [
    'convert_to_continuous'
]