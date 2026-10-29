import re
from dataclasses import dataclass
from re import Pattern
from typing import Any, Literal

import matplotlib.pyplot as plt
from patsy.desc import INTERCEPT
from scipy.stats._continuous_distns import _distn_names

from ._typing import LineStyle

__all__ = [
    'ANOVA',
    'CATEGORY',
    'COLOR',
    'DEFAULT',
    'DIST',
    'DOE',
    'KW',
    'LABEL',
    'LINE',
    'PERCENT_DECIMALS',
    'PLOTTER',
    'RE',
    'SIGMA_DIFFERENCE',
]


SIGMA_DIFFERENCE: float = 1.5
"""In order to make statements about long-term capabilities, the value 
of 1.5 is subtracted from the Sigma Level from short-term data. This 
value was determined empirically from manufacturing processes and takes 
into account that there are more influences on the process in the long 
term than in the short term. This value is the average of the short-term 
and long-term differences.

Sources
-------
Melzer, A. (2015) Six Sigma - Kompakt und praxisnah: 
Prozessverbesserung effizient und erfolgreich implementieren
(1. Aufl.). Springer Gabler, S. 107."""

PERCENT_DECIMALS = 2
"""Number of decimals used to format percentage values."""


@dataclass(frozen=True)
class _Line_:
    """Line style and width constants used throughout the DaSPi library.
    Don't instantiate this class directly. Use the `LINE` constant 
    instead."""

    WIDTH: float = 1.0
    """Line width for special lines e.g. StripeLines."""
    
    SOLID: LineStyle = (0, ())
    """Line style argument for a solid line."""
    
    DOTTED: LineStyle = (0, (2, 2))
    """Line style argument for a dotted line."""
    
    DASHED: LineStyle = (0, (5, 5))
    """Line style argument for a dashed line."""
    
    DASHDOT: LineStyle = (0, (8, 5, 2, 5))
    """Line style argument for a dashed dotted line."""

LINE = _Line_()
"""Line style and width constants used throughout the DaSPi library.

Examples
--------

```python
import matplotlib.pyplot as plt
import daspi as dsp

plt.plot([0, 1], [0, 1], **dsp.LINE.SOLID)
```
"""


@dataclass(frozen=True)
class _Kw_:
    """Keyword arguments for various plotting elements used throughout 
    the DaSPi library.
    
    Don't instantiate this class directly. Use the `KW` constant 
    instead."""

    _margin: float = 0.01
    """Margin between label and figure border."""

    @property
    def LINE(self) -> dict[str, Any]:
        """Base kwds for horizontal or vertical lines."""
        return {'lw': LINE.WIDTH, 'ls': LINE.DASHED}

    @property
    def HUE_HANDLES(self) -> dict[str, Any]:
        """Patch keyword arguments for genereting handles on 
        HueLabel."""
        return {'alpha': COLOR.FILL_ALPHA}

    @property
    def SHAPE_HANDLES(self) -> dict[str, Any]:
        """Line2D keyword arguments for genereting handles on 
        SizeLabel."""
        return {'xdata': [], 'ydata': [], 'color': COLOR.HANDLES, 'lw': 0}

    @property
    def SIZE_HANDLES(self) -> dict[str, Any]:
        """Line2D keyword arguments for genereting handles on 
        ShapeLabel."""
        return {
            'xdata': [], 'ydata': [], 'color': COLOR.HANDLES, 'marker': 'o', 
            'lw': 0, 'alpha': COLOR.MARKER_ALPHA}

    @property
    def CI_HANDLE(self) -> dict[str, Any]:
        """Keyword arguments for confidence interval handle."""
        return {'color': COLOR.HANDLES, 'alpha': COLOR.CI_ALPHA}

    @property
    def LEGEND(self) -> dict[str, Any]:
        """Figure legend at right side of figure."""
        return {
            'loc': 'upper right', 'alignment': 'left',
            'bbox_to_anchor': (1 - self._margin, 1 - self._margin)}

    @property
    def SAVE_CHART(self) -> dict[str, Any]:
        """Key word arguments for matplotlib savefig."""
        return {'bbox_inches': 'tight'}

    @property
    def XLABEL(self) -> dict[str, Any]:
        """Keyword arguments for Figure.text method used to add a 
        centered xlabel."""
        return {'x': 0.5, 'y': self._margin, 'ha': 'center', 'va': 'bottom'}

    @property
    def YLABEL(self) -> dict[str, Any]:
        """Keyword arguments for Figure.text method used to add a 
        centered xlabel."""
        return {
            'x': self._margin, 'y': 0.5, 'ha': 'left', 'va': 'center',
            'rotation': 90}

    @property
    def ROW_LABEL(self) -> dict[str, Any]:
        """Keyword Arguments for the Axes.text method used to add a 
        row label to each axes as text on LabelFacets."""
        return {
            'x': 1, 'y': 0.5, 'ha': 'left', 'va': 'center', 'rotation': -90}

    @property
    def ROW_TITLE(self) -> dict[str, Any]:
        """Keyword Arguments for the Axes.text method used to add a 
        row title to ax as text on LabelFacets."""
        return {**self.ROW_LABEL, 'x': 1 - self._margin, 'ha': 'right'}

    @property
    def COL_LABEL(self) -> dict[str, Any]:
        """Keyword Arguments for the Axes.text method used to add a 
        column label to each plot axis as text on LabelFacets."""
        return {'x': 0.5, 'y': 1, 'ha': 'center', 'va': 'bottom'}

    @property
    def COL_TITLE(self) -> dict[str, Any]:
        """Keyword Arguments for the Figure.text method used to add a 
        column title to each plot axis as text on LabelFacets."""
        return {**self.COL_LABEL, 'y': 1 - self._margin, 'va': 'top'}

    @property
    def FIG_TITLE(self) -> dict[str, Any]:
        """Keyword arguments for Figure.text method used for adding
        figure title on LabelFacets."""
        return {
            'x': self._margin, 'y': 1 - self._margin, 'ha': 'left',
            'va': 'top', 'size': 'x-large', 'in_layout': True}

    @property
    def SUB_TITLE(self) -> dict[str, Any]:
        """Keyword arguments for Figure.set_title method used for adding
        sub title at LabelFacets."""
        return {**self.FIG_TITLE, 'size': 'large'}

    @property
    def INFO(self) -> dict[str, Any]:
        """Adding info text at bottom left of figure."""
        return {
            'x': self._margin, 'y': self._margin, 'ha': 'left', 'va': 'bottom',
            'size': 'x-small', 'in_layout': True}

    @property
    def ERROR_BAR(self) -> dict[str, Any]:
        """Base keyword arguments for error bars."""
        return {'color': 'k', 'lw': 0.5, 'fmt': 'none'}

    @property
    def FIT_LINE(self) -> dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return {'zorder': 2.2, 'alpha': 0.8, 'marker': ''}

    @property
    def FIT_LINE_HANDLE(self) -> dict[str, Any]:
        """Keyword arguments for the legend handle of a fitted line
        (e.g. `MovingAverageLine`, `ExponentialSmoothingLine`)."""
        return {'lw': LINE.WIDTH, 'alpha': 0.8}

    @property
    def FIT_CI(self) -> dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return {'zorder': 2.3, 'alpha': COLOR.FILL_ALPHA, 'lw': 0}

    @property
    def LOWESS_CI(self) -> dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return self.FIT_CI

    @property
    def PRED_CI(self) -> dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return {
            'zorder': 2.1, 'alpha': COLOR.CI_ALPHA, 'lw': LINE.WIDTH,
            'ls': LINE.DASHED}

    @property
    def PROB_PC_FORMAT(self) -> dict[str, Any]:
        """Keyword arguments for percentage formatter used at 
        Probability Plotter."""
        return {'xmax': 1.0, 'decimals': None, 'symbol': '%'}

    @property
    def MEAN_LINE(self) -> dict[str, Any]:
        """Keyword arguments for mean line."""
        return {
            'lw': LINE.WIDTH, 'ls': LINE.DASHED, 'color': COLOR.MEAN,
            'zorder': 0.9}

    @property
    def MEDIAN_LINE(self) -> dict[str, Any]:
        """Keyword arguments for median line."""
        return {
            'lw': LINE.WIDTH, 'ls': LINE.DOTTED, 'color': COLOR.MEDIAN,
            'zorder': 0.8}

    @property
    def CONTROL_LINE(self) -> dict[str, Any]:
        """Keyword arguments for control limit line."""
        return {
            'lw': LINE.WIDTH, 'ls': LINE.SOLID, 'color': COLOR.PERCENTIL,
            'zorder': 0.7}

    @property
    def SPECIFICATION_LINE(self) -> dict[str, Any]:
        """Keyword arguments for specification limit line."""
        return {
            'lw': LINE.WIDTH, 'ls': LINE.SOLID, 'color': COLOR.BAD[:7],
            'zorder': 0.7}

    @property
    def STRIPES_CONFIDENCE(self) -> dict[str, Any]:
        """Keyword arguments for confidence area for stripes."""
        return {'alpha': COLOR.CI_ALPHA, 'lw': 0, 'zorder': 0.6}

    @property
    def PARETO_V(self) -> dict[str, Any]:
        """Keyword arguments for adding percentage texts in pareto chart.
        Use `PARETO_H` if `target_on_y' is False."""
        return {
            'x': 1, 'va': 'bottom', 'ha': 'right',
            'color': plt.rcParams['text.color'], 'fontsize': 'x-small',
            'zorder': 0.1}

    @property
    def PARETO_H(self) -> dict[str, Any]:
        """Keyword arguments for adding percentage texts in pareto chart.
        Use `PARETO_V` if `target_on_y' is True."""
        return {
            'y': 1, 'va': 'top', 'ha': 'left',
            'color': plt.rcParams['text.color'], 'fontsize': 'x-small',
            'rotation': -90, 'zorder': 0.1}

    @property
    def PARETO_LINE(self) -> dict[str, Any]:
        """Keyword arguments for plotting line in pareto chart."""
        return {
            'marker': plt.rcParams['scatter.marker'],
            'alpha': COLOR.MARKER_ALPHA}
    
KW = _Kw_()
"""Keyword arguments for various plotting elements used throughout the 
DaSPi library.

Example usage:

```python
import daspi as dsp
print(dsp.KW.MEAN_LINE)
```
"""


@dataclass(frozen=True)
class _Regex_:
    """Class containing precompiled regular expression patterns used 
    throughout the DaSPi library.
    
    Don't instantiate this class directly. Use the `RE` constant 
    instead."""

    ENCODED_NAME: Pattern = re.compile(r'(.+)\[[T.]?.+\]')
    """Patsy encoded column name."""
    
    ENCODED_VALUE: Pattern = re.compile(r'\[T.(.+)]')
    """Patsy encoded value shown in parameter name."""

RE = _Regex_()
"""Precompiled regular expression patterns used throughout the DaSPi 
library."""


@dataclass(frozen=True)
class _Color_:
    """Color constants used throughout the DaSPi library.
    
    Don't instantiate this class directly. Use the `COLOR` constant 
    instead."""

    GOOD: str = '#2ca02c7f'
    """Color for things that should be represented as 'good'."""
    
    BAD: str = '#d627287f'
    """Color for things that should be represented as 'bad'."""
    
    ANOMALY: str = '#ff7f0e7f'
    """Color for anomalies."""
    
    SPECIAL_LINE: str = "#0e8bff7e"
    """Color for fitted lines or thresholds in precas charts."""
    
    MEAN: str = '#2F3333'
    """Color for mean line used for StripesFacets."""
    
    MEDIAN: str = '#626666'
    """Color for median line used for StripesFacets."""
    
    PERCENTIL: str = '#979A9A'
    """Color for upper and lower percentil line used for StripesFacets."""
    
    HANDLES: str = '#626666'
    """Color for size and shape legend handles."""
    
    STRIPE: str = '#CACCCC'
    """Color for individual stripes."""
    
    TRANSPARENT: str = '#ffffff00'
    """Transparent 'color' to hide ticks or other stuff."""
    
    BLUR: str = '#ffffffaa'
    """Color to blur other colors, by adding 2/3 white."""
    
    DARKEN: str = '#00000019'
    """Color to darken other colors, by adding 10 % black."""
    
    MARKER_ALPHA: float = 0.5
    """The covering capacity of markers."""
    
    FILL_ALPHA: float = 0.5
    """The covering capacity of filled areas."""
    
    CI_ALPHA: float = 0.2
    """The covering capacity of confidence intervals."""
    
    CI_STRIPES: str = '#D7DBDF'
    """Color for confidence interval stripes."""
    
    SPECIAL_AREA1: str = "#145a5aa0"
    """Color for special area 1."""
    
    SPECIAL_AREA2: str = "#5a47149f"
    """Color for special area 2."""

    @property
    def PALETTE(self) -> list[str]:
        """Color palette from the current matplotlib rcParams property 
        cycle."""
        return plt.rcParams['axes.prop_cycle'].by_key()['color']

    @property
    def LIMITS(self) -> tuple[str, str]:
        """Color for specification limits. It returns a tuple containing 
        the colors for the lower and upper specification limits, 
        respectively, and is defined by the constant :attr:`BAD` 
        (twice)."""
        return (self.BAD, self.BAD)

    @property
    def STATISTIC_LINES(self) -> tuple[str, str, str]:
        """Statistic lines color in order lower (Percentil Q_0.99865)
        upper (Percentil Q_0.00135) and mean.
        
        It returns a tuple containing the colors for the lower 
        percentile line, upper percentile line, and mean line, 
        respectively and is defined by the constants :attr:`PERCENTIL` 
        and :attr:`MEAN`.
        """
        return (self.PERCENTIL, self.PERCENTIL, self.MEAN)
    
COLOR = _Color_()
"""Color constants for the plotting library. Used to maintain consistent 
color schemes across different plots.

Examples
--------
Getting the color palette:
```python
import daspi as dsp

print(dsp.COLOR.PALETTE)
```

Use individual color constants for a Stripe within a plot:
```python
import numpy as np
import daspi as dsp

x = np.array([1, 2, 3, 4, 5])  # Example data for demonstration purposes
x_bar = np.mean(x)

stripe = dsp.StripeLine(
    label=r'\bar x = 'f'{x_bar:.2f}',
    position=x_bar,
    color=dsp.COLOR.SPECIAL_LINE,)
```
"""


@dataclass(frozen=True)
class _Label_:
    """Class containing label-related constants used throughout the 
    DaSPi library. Don't instantiate this class directly. Use the 
    `LABEL` constant instead."""

    PPI: float = 1/72
    """Font size points per inch."""
    PADDING: int = 6
    """Additional padding in pixels between labels e.g. title and
    subtitle."""
    X_ALIGNEMENT: int = 35
    """Horizontal alignment of labels (fig title, sub title and info), 
    given in pixels from the left figure border."""
    
LABEL = _Label_()
"""Label-related constants used throughout the DaSPi library, such as 
font size conversion, padding between labels and horizontal alignment 
of figure titles."""


@dataclass(frozen=True)
class _Plotter_:
    """Column name constants used internally by the plotter classes to 
    store generated or transformed data (e.g. fitted values, confidence 
    intervals, LOWESS results).
    
    Don't instantiate this class directly. Use the `PLOTTER` constant 
    instead."""

    FEATURE: Literal['_feature_'] = '_feature_'
    """Column name for generated features."""
    
    TRANSFORMED_FEATURE: Literal['_transformed_'] = '_transformed_'
    """Column name for transformed feature if no features were given 
    during initialization."""
    
    F_BASE_NAME: Literal['_base_'] = '_base_'
    """Column name for the base position of transformed features such 
    as Jitter or KDE."""
    
    LOWER: Literal['_lower_'] = '_lower_'
    """Column name for lower points used for e.g. error bars."""
    
    UPPER: Literal['_upper_'] = '_upper_'
    """Column name for upper points used for e.g. error bars."""
    
    FITTED_VALUES: Literal['_fitted_values_'] = '_fitted_values_'
    """Column name for fitted values."""
    
    FIT_CI_LOW: Literal['_fit_ci_low_'] = '_fit_ci_low_'
    """Column name for lower confidence of fitted values."""
    
    FIT_CI_UPP: Literal['_fit_ci_upp_'] = '_fit_ci_upp_'
    """Column name for upper confidence of fitted values."""
    
    PRED_CI_LOW: Literal['_pred_ci_low_'] = '_pred_ci_low_'
    """Column name for lower confidence of fitted line."""
    
    PRED_CI_UPP: Literal['_pred_ci_upp_'] = '_pred_ci_upp_'
    """Column name for upper confidence of fitted line."""
    
    LOWESS_TARGET: Literal['lowess_target'] = 'lowess_target'
    """Column name for lowess target values."""
    
    LOWESS_FEATURE: Literal['lowess_feature'] = 'lowess_feature'
    """Column name for lowess feature values."""
    
    LOWESS_LOW: Literal['lowess_low'] = 'lowess_low'
    """Column name for lower confidence of fitted values."""
    
    LOWESS_UPP: Literal['lowess_upp'] = 'lowess_upp'
    """Column name for upper confidence of fitted values."""
    
    PARETO_N_TICKS: Literal[11] = 11
    """Number of ticks to represent percentage values in Pareto charts."""
    
    PARETO_AXLIM_FACTOR: float = 1.05
    """Factor to set the axis limit for the upper value in the Pareto
    diagram."""
    
    PARETO_F_MARGIN: float = 0.1
    """Margin in feature axis direction to ensure space for percentage
    values."""
    
    SUBGROUP: Literal['_subgroup_'] = '_subgroup_'
    """Column name for subgroups used mostly in TransformPlotter."""

    SMOOTHED_TARGET: Literal['_smoothed_target_'] = '_smoothed_target_'
    """Column name for smoothed target values (moving average,
    exponential smoothing)."""

    SMOOTHED_FEATURE: Literal['_smoothed_feature_'] = '_smoothed_feature_'
    """Column name for the feature values matching the smoothed target
    values (moving average, exponential smoothing)."""

    @property
    def REGRESSION_CI_NAMES(self) -> tuple[str, str, str, str]:
        """Get names for regression confidences in order:
        
            - lower confidence level of fitted values
            - upper confidence level of fitted values
            - lower confidence level of predicted values
            - upper confidence level of predicted values"""
        names = (
            self.FIT_CI_LOW, self.FIT_CI_UPP,
            self.PRED_CI_LOW, self.PRED_CI_UPP)
        return names
    
PLOTTER = _Plotter_()
"""Column name constants used internally by the plotter classes to 
store generated or transformed data."""


@dataclass(frozen=True)
class _Category_:
    """Constants used to visually distinguish categorical data (hue, 
    size and shape) in plots.
    
    Don't instantiate this class directly. Use the `CATEGORY` constant 
    instead."""

    MARKERS: tuple[str, ...] = ('o', 's', '^', 'p', 'D', 'v', 'P', 'X', '*')
    """Markers for visually easy distinction."""
    
    MARKERSIZE_LIMITS: tuple[int, int] = (1, 13)
    """Marker size limits used for sizing."""
    
    N_SIZE_BINS: int = 5
    """Amount of size bins in the legend."""
    
    FEATURE_SPACE: float = 0.8
    """Space required in the direction of the feature axis to draw
    categorical artists such as violin or jitter."""
    
    FEATURE_PAD: float = 0.05
    """Padding between dodget artists."""

    @property
    def SIZE_LIMITS(self) -> tuple[int, int]:
        """Used for scatter plots. The area must be specified there 
        instead of the height, as with markers in line plots.
        See: https://stackoverflow.com/a/14860958/11362192"""
        return (self.MARKERSIZE_LIMITS[0]**2, self.MARKERSIZE_LIMITS[1]**2)

    @property
    def PALETTE(self) -> tuple[str, ...]:
        """Get the current color palette as tuple."""
        return tuple(COLOR.PALETTE)
    
CATEGORY = _Category_()
"""Constants used to visually distinguish categorical data (hue, size 
and shape) in plots."""


@dataclass(frozen=True)
class _Distribution_:
    """Groups of Scipy continuous distribution names used for fitting 
    and selecting distributions throughout the DaSPi library.
    
    Don't instantiate this class directly. Use the `DIST` constant 
    instead."""
    
    POSSIBLE: tuple[str, ...] = tuple(
        d for d in _distn_names 
        if d not in ('levy_stable', 'studentized_range'))
    """Get all possible continuous distributions from Scipy, except 
    'levy_stable' and 'studentized_range'. These distributions are often 
    problematic or have special requirements, so they are excluded."""
    
    COMMON: tuple[str, ...] = (
        'norm',
        'chi2',
        'foldnorm',
        'rayleigh',
        'weibull_min',
        'gamma',
        'wald',
        'expon',
        'logistic',
        'lognorm',)
    """Distributions that commonly occur in practice. These are often a 
    good starting point for fitting. It includes the following 
    distributions:
    - norm
    - chi2
    - foldnorm
    - rayleigh
    - weibull_min
    - gamma
    - wald
    - expon
    - logistic
    - lognorm
    """

    COMMON_NOT_NORM: tuple[str, ...] = tuple(d for d in COMMON if d != 'norm')
    """Get all common distributions but norm, usefull to fit if norm
    tests fail. It includes the following distributions:
    - chi2
    - foldnorm
    - rayleigh
    - weibull_min
    - gamma
    - wald
    - expon
    - logistic
    - lognorm
    """

    POSITIVE: tuple[str, ...] = (
        'chi2',
        'rayleigh',
        'expon',
        'gamma',
        'weibull_min',
        'lognorm',
        'invgauss',)
    """Continuous distributions with support x >= 0. Those are useful 
    when modeling quantities that cannot be negative. It includes the
    following distributions:
    - chi2
    - rayleigh
    - expon
    - gamma
    - weibull_min
    - lognorm
    - invgauss
    """

    POSITIVE_SKEWED: tuple[str, ...] = (
        'expon',
        'gamma',
        'weibull_min',
        'lognorm',
        'invgauss',
        'rayleigh',)
    """Common positive and right-skewed distributions. These are often 
    used when modeling quantities that are strictly positive and exhibit
    right skewness. It includes the following distributions:
    - expon
    - gamma
    - weibull_min
    - lognorm
    - invgauss
    - rayleigh
    """

    SYMMETRIC: tuple[str, ...] = (
        'norm',
        'logistic',)
    """Common symmetric distributions. These distributions have a 
    symmetric shape around their mean. It includes the following 
    distributions:
    - norm
    - logistic
    """

    LIFETIME: tuple[str, ...] = (
        'expon',
        'gamma',
        'weibull_min',
        'lognorm',
        'invgauss',)
    """Common lifetime and reliability distributions. These 
    distributions are often used to model the time until an event 
    occurs, such as failure or death. It includes the following 
    distributions:
    - expon
    - gamma
    - weibull_min
    - lognorm
    - invgauss
    """
   
    FAST_FIT: tuple[str, ...] = (
        'norm',
        'gamma',
        'weibull_min',
        'lognorm',)
    """Distributions that are fast to fit and cover surprisingly many 
    real datasets. These distributions are typically chosen when a 
    balance between computational efficiency and modeling flexibility is 
    desired. It includes the following distributions:
    - norm
    - gamma
    - weibull_min
    - lognorm
    """

    @property
    def UNCERTAINTY_FACTORS(self) -> dict[str, float]:
        """Factors used to calculate the measurement uncertainty 
        (equivalent to the standard deviation) of a distribution using 
        the error limit. 
        
        If you have a rectangular distribution, such as the resolution, 
        the entire width is specified. With uncertainties, the scatter 
        is always specified in only one direction. Therefore, the 
        resolution is first divided by the coverage factor (k = 2), then 
        by the rectangular factor, i.e., 1/(2 × √3). In the literature, 
        you often see 1/√12, which is the same (if k = 2).
        
        Sources
        -------
        - https://de.wikipedia.org/wiki/Stetige_Gleichverteilung
        - https://www.versuchsmethoden.de/Mess-System-Analyse.pdf"""
        return {
            'rectangular': 3**0.5,  # ≈ 1.732
            'triangular': 2.0,
            'normal': 1.0}
    
DIST = _Distribution_()
"""Groups of Scipy continuous distribution names used for fitting and 
selecting distributions throughout the DaSPi library."""


@dataclass(frozen=True)
class _Default_:
    """Default values used throughout the DaSPi library, e.g. confidence 
    level, sequence lengths for kernel density estimation and LOWESS, 
    and quantile ranges.
    
    Don't instantiate this class directly. Use the `DEFAULT` constant 
    instead."""

    CONFIDENCE_LEVEL: float = 0.95
    """Default confidence level. 95 % confidence interval is used very 
    frequently in practice."""
    
    FEATURE_BASE: Literal[0] = 0
    """Default feature base position for e.g. Jitter or KDE."""
    
    KD_SEQUENCE_LEN: Literal[300] = 300
    """Amount of points for kernel density sequence."""
    
    LOWESS_SEQUENCE_LEN: Literal[300] = 300
    """Amount of points for LOWESS curve."""
    
    PLOTTING_COLOR: str = '#2F3333'
    """Default color in plots if there is no hueing only used for styles
    they dont use the regular color palette order (e.g. daspi and 
    ggplot2)."""
    
    QUANTILE_RANGES: tuple[float, float, float] = (0.682, 0.954, 0.997)
    """Default quantile ranges for the Quantile Plotter.

    These values represent the quantiles corresponding to the standard 
    deviations (σ) of a normal distribution:
    - 0.682 = approximately ±1 σ, capturing about 68.2% of the data.
    - 0.954 = approximately ±2 σ, capturing about 95.4% of the data.
    - 0.997 = approximately ±3 σ, capturing about 99.7% of the data.
    """
    AGREEMENTS: tuple[int, int, int] = (2, 4, 6)
    """Default agreement levels for the Quantile Plotter and for the
    quantiles in GaussianKDE and Violin Plotter.
    
    These values represent the quantiles corresponding to the standard
    deviations (σ) of a normal distribution:
    - 2 = ±1 σ, capturing about 68.2% of the data.
    - 4 = ±2 σ, capturing about 95.4% of the data.
    - 6 = ±3 σ, capturing about 99.7% of the data.
    """

    MA_WINDOW: Literal[5] = 5
    """Default window size for the `MovingAverageLine` Plotter."""

    EWM_ALPHA: float = 0.3
    """Default smoothing factor (alpha) for the
    `ExponentialSmoothingLine` Plotter."""

    @property
    def MARKER(self) -> str:
        """Default marker for scatter plots."""
        marker = plt.rcParams['lines.marker']
        if marker == 'None':
            marker = CATEGORY.MARKERS[0]
        return marker
    
DEFAULT = _Default_()
"""Default values used throughout the DaSPi library, e.g. confidence 
level, sequence lengths for kernel density estimation and LOWESS, and 
quantile ranges."""


@dataclass(frozen=True)
class _Anova_:
    """Column and row names used when building ANOVA, VIF, Gage R&R and 
    uncertainty tables throughout the DaSPi library.
    
    Don't instantiate this class directly. Use the `ANOVA` constant 
    instead."""

    SEP: Literal[':'] = ':'
    """Separator joining the names of interacting factors into a single 
    column name, e.g. 'A:B' for the interaction between factors 'A' and 
    'B'."""
    
    INTERCEPT: str = INTERCEPT.name()
    """Column name used by patsy for the intercept term of a linear 
    model."""
    
    EFFECTS: Literal['effects'] = 'effects'
    """Default column name for the effect names (factors and their 
    interactions) in a model's results."""
    
    TARGET: Literal['target'] = 'target'
    """Default column name for the target (measured) values used in 
    `GageStudyModel`."""
    
    FACTORS: Literal['factors'] = 'factors'
    """Default column name for the factor values used when building a 
    design matrix."""
    
    REFERENCE: Literal['reference'] = 'reference'
    """Default column name for the reference (true or nominal) values 
    used in `GageStudyModel`."""
    
    SOURCE: Literal['Source'] = 'Source'
    """Default column name for the source of variation (e.g. factor or 
    residual name) in an ANOVA table."""
    
    VIF: Literal['VIF'] = 'VIF'
    """Default column name for the Variance Inflation Factor values used 
    to detect multicollinearity in an ANOVA table."""
    
    EPOCH: Literal['epoch'] = 'epoch'
    """Column name for the step at which a term was added or removed 
    during recursive elimination in a goodness-of-fit table, see 
    `LinearModel.gof_metrics`."""
    
    FORMULA: Literal['formula'] = 'formula'
    """Column name for the model formula at a given elimination step in 
    a goodness-of-fit table, see `LinearModel.gof_metrics` and 
    `LinearModel.formula_at_epoch`."""
    
    HIERARCHICAL: Literal['hierarchical'] = 'hierarchical'
    """Column name for the flag indicating whether the model is 
    hierarchical in a goodness-of-fit table, see 
    `LinearModel.gof_metrics`."""
    
    LEAST_PARAMETER: Literal['least_parameter'] = 'least_parameter'
    """Column name for the least significant term in a goodness-of-fit 
    table, see `LinearModel.gof_metrics`."""
    
    P_LEAST: Literal['p_least'] = 'p_least'
    """Column name for the p-value of the least significant term in a 
    goodness-of-fit table, see `LinearModel.gof_metrics`."""
    
    AIC: Literal['aic'] = 'aic'
    """Column name for Akaike's information criterion in a 
    goodness-of-fit table, see `LinearModel.gof_metrics`."""
    
    R2: Literal['r2'] = 'r2'
    """Column name for R-squared in a goodness-of-fit table, see 
    `LinearModel.gof_metrics`."""
    
    R2_ADJ: Literal['r2_adj'] = 'r2_adj'
    """Column name for adjusted R-squared in a goodness-of-fit table, 
    see `LinearModel.gof_metrics`."""
    
    R2_PRED: Literal['r2_pred'] = 'r2_pred'
    """Column name for predicted R-squared in a goodness-of-fit table, 
    see `LinearModel.gof_metrics`."""
    
    SMALLEST_INTERACTION_ORDER: int = 2
    """Smallest possible interaction order, i.e. the interaction between 
    two factors (A:B)."""
    
    RESIDUAL: Literal['Residual'] = 'Residual'
    """Name in anova table for residual (not explained) values."""
    
    OBSERVATION: Literal['Observation'] = 'Observation'
    """Name in residual table for observation order."""
    
    PREDICTION: Literal['Prediction'] = 'Prediction'
    """Name in residual table for predicted values."""
    
    TOTAL: Literal['Total'] = 'Total'
    """Name in anova table for total (sum of the others) values."""
    
    PV: Literal['PV'] = 'PV'
    """Name in Gage R&R table for Part Variation, i.e. the variation 
    coming from differences between the measured parts."""
    
    EV: Literal['EV'] = 'EV'
    """Name in Gage R&R table for Equipment Variation (repeatability), 
    i.e. the variation coming from the measurement equipment itself."""
    
    AV: Literal['AV'] = 'AV'
    """Name in Gage R&R table for Appraiser Variation (reproducibility), 
    i.e. the variation coming from different appraisers/operators."""
    
    GV: Literal['GV'] = 'GV'
    """Name in Gage R&R table for Gage Variation, i.e. the combination of 
    `EV` and `AV`."""
    
    IA: Literal['IA'] = 'IA'
    """Name in R&R table for the interaction between appraiser and part 
    values."""
    
    RNR: Literal['R&R'] = 'R&R'
    """Name in R&R table for the total Gage R&R value, i.e. the sum of 
    `GV` and `PV`."""

    @property
    def TABLE_COLNAMES(self) -> list[str]:
        """Column names when crating the anova table using LinearModel 
        class"""
        return ['DF', 'SS', 'MS', 'F', 'p', 'n2']
    
    @property
    def VIF_COLNAMES(self) -> list[str]:
        """Column names when crating the vif table using the 
        `variance_inflation_factor` function."""
        return ['DF', self.VIF, 'GVIF', 'Threshold', 'Collinear', 'Method']
    
    @property
    def RNR_COLNAMES(self) -> list[str]:
        """Column names when crating the rnr table using the 
        `variance_inflation_factor` function."""
        return ['MS', 'MS/Total', 's', '6s', '6s/Total', '6s/Tolerance']
    
    @property
    def CAPABILITY_COLNAMES(self) -> list[str]:
        """Column names when crating the capability table using the 
        `capability` method of GageStudyModel."""
        return ['Value', 'Limit', 'Capable', 'T_min']
    
    @property
    def CAPABILITY_ROWS(self) -> list[str]:
        """Row names (indices) when crating the capability table using
        the `capability` method of GageStudyModel."""
        return ['Cg', 'Cgk', 'RE_ratio', 'Q_MS', 'p_BI']

    @property
    def UNCERTAINTY_COLNAMES(self) -> list[str]:
        """Column names when crating the uncertainty table using the 
        `uncertainties` method of GageRnRModel."""
        return ['u', 'U', 'Q', 'rank']

    @property
    def UNCERTAINTY_ROWS_MS(self) -> list[str]:
        """Row names (indices) when crating the uncertainty table using
        the `uncertainties` method of GageStudyModel."""
        return ['CAL', 'RE', 'BI', 'LIN', 'EVR', 'REST', 'MS']

    @property
    def UNCERTAINTY_ROWS_MP(self) -> list[str]:
        """Row names (indices) when crating the uncertainty table using
        the `uncertainties` method of GageRnRModel."""
        return ['EVO', 'AV', 'GV', 'IA', 'T', 'STAB', 'OBJ', 'REST', 'MP']
    
    @property
    def REFERENCE_ANALYSIS_COLNAMES(self) -> list[str]:
        """Column names when crating the reference analysis table using
        the `reference_analysis` method of GageRnRModel."""
        return ['Ref',	'mean', 'Bias', 's', 'R']

ANOVA = _Anova_()
"""Column and row names used when building ANOVA, VIF, Gage R&R and 
uncertainty tables throughout the DaSPi library."""


@dataclass(frozen=True)
class _DOE_:
    """Design of Experiments (DOE) constants."""

    STD_ORDER: Literal['std_order'] = 'std_order'
    """Name for standard order in the design space."""
    
    RUN_ORDER: Literal['run_order'] = 'run_order'
    """Name for run order in the design space."""
    
    CENTRAL_POINT: Literal['central_point'] = 'central_point'
    """Name for central points in the design space."""
    
    BLOCK: Literal['block'] = 'block'
    """Name for block in the design space."""
    
    REPLICA: Literal['replica'] = 'replica'
    """Name for replicate in the design space."""
    
    CORRECTED_CENTRAL: int = 0
    """Coded value for central points in the design space."""

DOE = _DOE_()
"""Design of Experiments (DOE) constants used for column names and 
coded values in the design space."""
