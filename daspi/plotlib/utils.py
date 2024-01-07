import numpy as np

from scipy import stats
from typing import Tuple
from numpy.typing import ArrayLike

from .constants import KDE_POINTS

def estimate_kernel_density(
        data: ArrayLike, scale: float | None = None, n_points: int = KDE_POINTS
    ) -> Tuple[ArrayLike, ArrayLike]:
    """Estimates the kernel density of data and returns values that are 
    useful for a plot. If those values are plotted in combination with 
    a histogram, set scale as max value of the hostogram.
    
    Kernel density estimation is a way to estimate the probability 
    density function (PDF) of a random variable in a non-parametric way. 
    The used gaussian_kde function of scipy.stats works for both 
    uni-variate and multi-variate data. It includes automatic bandwidth 
    determination. The estimation works best for a unimodal 
    distribution; bimodal or multi-modal distributions tend to be 
    oversmoothed.
    
    Parameters
    ----------
    data : array_like
        1-D array of datapoints to estimate from.
    scale : float or None, optional
        If the KDE curve is plotted in combination with other data 
        (e.g. a histogram), you can use scale to specify the height at 
        the maximum point of the KDE curve. If this value is specified, 
        the area under the curve will not be normalized, by default None
    n_points : int, optional
        Number of points the estimation and sequence should have,
        by default KDE_POINTS (defined in constants.py)

    Returns
    -------
    sequence : 1D array
        Data points at regular intervals from input data minimum to 
        maximum
    estimation : 1D array
        Data points of kernel density estimation
    """
    data = np.array(data)[~np.isnan(data)]
    sequence = np.linspace(data.min(), data.max(), n_points)
    estimation = stats.gaussian_kde(data, bw_method='scott')(sequence)
    if scale is not None:
        estimation = estimation*scale/estimation.max()
    return sequence, estimation

__all__ = [
    estimate_kernel_density.__name__,
]