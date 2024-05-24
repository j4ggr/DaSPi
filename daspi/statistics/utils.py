import numpy as np

from scipy import stats
from scipy.stats import rv_continuous

from typing import Any
from typing import Sequence
from typing import Generator
from numpy.typing import NDArray


def chunker(
        samples: Sequence[Any] | NDArray[Any], n_sections: int
        ) -> Generator[NDArray, Any, None]:
    """Divides the data into a specified number of sections.
    
    Parameters
    ----------
    sample : Sequence[Any]
        A one-dimensional array-like object containing the samples.
    n_sections : int
        Amount of sections to divide the data into.
        
    Yields
    ------
    NDArray
        A section of the data.
    
    Notes
    -----
    If equal-sized sections cannot be created, the first sections are 
    one larger than the rest.

    If more sections are to be created than the number of samples, 
    empty arrays are created.
    """
    assert n_sections > 0 and isinstance(n_sections, int)
    size, extras = divmod(len(samples), n_sections)
    sizes = extras*[size + 1] + (n_sections - extras)*[size]
    slicing_positions = np.array([0] + sizes).cumsum()

    _samples = np.asarray(samples)
    for i in range(n_sections):
        yield _samples[slicing_positions[i]:slicing_positions[i+1]]

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
    'chunker',
    'convert_to_continuous'
]