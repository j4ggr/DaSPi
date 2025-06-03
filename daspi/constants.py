import re
import matplotlib.pyplot as plt

from re import Pattern
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from patsy.desc import INTERCEPT
from dataclasses import dataclass
from scipy.stats._continuous_distns import _distn_names

from ._typing import LineStyle


SIGMA_DIFFERENCE: float = 1.5
"""In order to make statements about long-term capabilities, the value 
of 1.5 is subtracted from the Sigma level from short-term data. This 
value was determined empirically from manufacturing processes and takes 
into account that there are more influences on the process in the long 
term than in the short term. This value is the average of the short-term 
and long-term differences."""


@dataclass(frozen=True)
class _Line_:
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


@dataclass(frozen=True)
class _Kw_:

    _margin: float = 0.01
    """Margin between label and figure border."""

    @property
    def LINE(self) -> Dict[str, Any]:
        """Base kwds for horizontal or vertical lines."""
        return dict(lw=LINE.WIDTH, ls=LINE.DASHED)

    @property
    def HUE_HANDLES(self) -> Dict[str, Any]:
        """Patch keyword arguments for genereting handles on 
        HueLabel."""
        return dict(alpha=COLOR.FILL_ALPHA)

    @property
    def SHAPE_HANDLES(self) -> Dict[str, Any]:
        """Line2D keyword arguments for genereting handles on 
        SizeLabel."""
        return dict(xdata=[], ydata=[], color=COLOR.HANDLES, lw=0)

    @property
    def SIZE_HANDLES(self) -> Dict[str, Any]:
        """Line2D keyword arguments for genereting handles on 
        ShapeLabel."""
        return dict(
            xdata=[], ydata=[], color=COLOR.HANDLES, marker='o', lw=0, 
            alpha=COLOR.MARKER_ALPHA)

    @property
    def CI_HANDLE(self) -> Dict[str, Any]:
        """Keyword arguments for confidence interval handle."""
        return dict(color=COLOR.HANDLES, alpha=COLOR.CI_ALPHA)

    @property
    def LEGEND(self) -> Dict[str, Any]:
        """Figure legend at right side of figure."""
        return dict(
            loc='upper right', alignment='left',
            bbox_to_anchor=(1 - self._margin, 1 - self._margin))

    @property
    def SAVE_CHART(self) -> Dict[str, Any]:
        """Key word arguments for matplotlib savefig."""
        return dict(bbox_inches='tight')

    @property
    def XLABEL(self) -> Dict[str, Any]:
        """Keyword arguments for Figure.text method used to add a 
        centered xlabel."""
        return dict(
            x=0.5, y=self._margin, ha='center', va='bottom')

    @property
    def YLABEL(self) -> Dict[str, Any]:
        """Keyword arguments for Figure.text method used to add a 
        centered xlabel."""
        return dict(
            x=self._margin, y=0.5, ha='left', va='center', rotation=90)

    @property
    def ROW_LABEL(self) -> Dict[str, Any]:
        """Keyword Arguments for the Axes.text method used to add a 
        row label to each axes as text on LabelFacets."""
        return dict(
            x=1, y=0.5, ha='left', va='center', rotation=-90)

    @property
    def ROW_TITLE(self) -> Dict[str, Any]:
        """Keyword Arguments for the Axes.text method used to add a 
        row title to ax as text on LabelFacets."""
        return self.ROW_LABEL | dict(x=1 - self._margin, ha='right')

    @property
    def COL_LABEL(self) -> Dict[str, Any]:
        """Keyword Arguments for the Axes.text method used to add a 
        column label to each plot axis as text on LabelFacets."""
        return dict(x=0.5, y=1, ha='center', va='bottom')

    @property
    def COL_TITLE(self) -> Dict[str, Any]:
        """Keyword Arguments for the Figure.text method used to add a 
        column title to each plot axis as text on LabelFacets."""
        return self.COL_LABEL | dict(y=1 - self._margin, va='top')

    @property
    def FIG_TITLE(self) -> Dict[str, Any]:
        """Keyword arguments for Figure.text method used for adding
        figure title on LabelFacets."""
        return dict(
            x=self._margin, y=1 - self._margin, ha='left', va='top',
            size='x-large', in_layout=True)

    @property
    def SUB_TITLE(self) -> Dict[str, Any]:
        """Keyword arguments for Figure.set_title method used for adding
        sub title at LabelFacets."""
        return self.FIG_TITLE | dict(size='large')

    @property
    def INFO(self) -> Dict[str, Any]:
        """Adding info text at bottom left of figure."""
        return dict(
            x=self._margin, y=self._margin, ha='left', va='bottom',
            size='x-small', in_layout=True)

    @property
    def ERROR_BAR(self) -> Dict[str, Any]:
        """Base keyword arguments for error bars."""
        return dict(color='k', lw=0.5, fmt='none')

    @property
    def FIT_LINE(self) -> Dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return dict(zorder=2.2, alpha=0.8, marker='')

    @property
    def FIT_CI(self) -> Dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return dict(zorder=2.3, alpha=COLOR.FILL_ALPHA, lw=0)

    @property
    def LOWESS_CI(self) -> Dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return self.FIT_CI

    @property
    def PRED_CI(self) -> Dict[str, Any]:
        """Keyword arguments for confidence interval area for fit."""
        return dict(
            zorder=2.1, alpha=COLOR.CI_ALPHA, lw=LINE.WIDTH, ls=LINE.DASHED)

    @property
    def PROB_PC_FORMAT(self) -> Dict[str, Any]:
        """Keyword arguments for percentage formatter used at 
        Probability Plotter."""
        return dict(xmax=1.0, decimals=None, symbol='%')

    @property
    def MEAN_LINE(self) -> Dict[str, Any]:
        """Keyword arguments for mean line."""
        return dict(lw=LINE.WIDTH, ls=LINE.DASHED, color=COLOR.MEAN, zorder=0.9)

    @property
    def MEDIAN_LINE(self) -> Dict[str, Any]:
        """Keyword arguments for median line."""
        return dict(
            lw=LINE.WIDTH, ls=LINE.DOTTED, color=COLOR.MEDIAN, zorder=0.8)

    @property
    def CONTROL_LINE(self) -> Dict[str, Any]:
        """Keyword arguments for control limit line."""
        return dict(
            lw=LINE.WIDTH, ls=LINE.SOLID, color=COLOR.PERCENTIL, zorder=0.7)

    @property
    def SPECIFICATION_LINE(self) -> Dict[str, Any]:
        """Keyword arguments for specification limit line."""
        return dict(
            lw=LINE.WIDTH, ls=LINE.SOLID, color=COLOR.BAD[:7], zorder=0.7)

    @property
    def STRIPES_CONFIDENCE(self) -> Dict[str, Any]:
        """Keyword arguments for confidence area for stripes."""
        return dict(alpha=COLOR.CI_ALPHA, lw=0, zorder=0.6)

    @property
    def PARETO_V(self) -> Dict[str, Any]:
        """Keyword arguments for adding percentage texts in pareto chart.
        Use `PARETO_H` if `target_on_y' is False."""
        return dict(
            x=1, va='bottom', ha='right', color=plt.rcParams['text.color'],
            fontsize='x-small', zorder=0.1)

    @property
    def PARETO_H(self) -> Dict[str, Any]:
        """Keyword arguments for adding percentage texts in pareto chart.
        Use `PARETO_V` if `target_on_y' is True."""
        return dict(
            y=1, va='top', ha='left', color=plt.rcParams['text.color'],
            fontsize='x-small', rotation=-90, zorder=0.1)

    @property
    def PARETO_LINE(self) -> Dict[str, Any]:
        """Keyword arguments for plotting line in pareto chart."""
        return dict(
            marker=plt.rcParams['scatter.marker'], alpha=COLOR.MARKER_ALPHA)
    
KW = _Kw_()


@dataclass(frozen=True)
class _Regex_:

    ENCODED_NAME: Pattern = re.compile(r'(.+)\[[T.]?.+\]')
    """Patsy encoded column name."""
    ENCODED_VALUE: Pattern = re.compile(r'\[T.(.+)]')
    """Patsy encoded value shown in parameter name."""

RE = _Regex_()


@dataclass(frozen=True)
class _Color_:

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
    def PALETTE(self) -> List[str]:
        """Get prop cycler color palette."""
        return plt.rcParams['axes.prop_cycle'].by_key()['color']

    @property
    def LIMITS(self) -> Tuple[str, str]:
        """Color for specification limits."""
        return (self.BAD, self.BAD)

    @property
    def STATISTIC_LINES(self) -> Tuple[str, str, str]:
        """Statistic lines color in order lower (Percentil Q_0.99865)
        upper (Percentil Q_0.00135) and mean."""
        return (self.PERCENTIL, self.PERCENTIL, self.MEAN)
    
COLOR = _Color_()


@dataclass(frozen=True)
class _Label_:

    PPI: float = 1/72
    """Font size points per inch."""
    PADDING: int = 6
    """Additional padding in pixels between labels e.g. title and
    subtitle."""
    X_ALIGNEMENT: int = 35
    """Horizontal alignment of labels (fig title, sub title and info)."""
    
LABEL = _Label_()


@dataclass(frozen=True)
class _Plotter_:

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

    @property
    def REGRESSION_CI_NAMES(self) -> Tuple[str, str, str, str]:
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


@dataclass(frozen=True)
class _Category_:

    MARKERS: Tuple[str, ...] = ('o', 's', '^', 'p', 'D', 'v', 'P', 'X', '*')
    """Markers for visually easy distinction."""
    MARKERSIZE_LIMITS: Tuple[int, int] = (1, 13)
    """Marker size limits used for sizing."""
    N_SIZE_BINS: int = 5
    """Amount of size bins in the legend."""
    FEATURE_SPACE: float = 0.8
    """Space required in the direction of the feature axis to draw
    categorical artists such as violin or jitter."""
    FEATURE_PAD: float = 0.05
    """Padding between dodget artists."""

    @property
    def SIZE_LIMITS(self) -> Tuple[int, int]:
        """Used for scatter plots. The area must be specified there 
        instead of the height, as with markers in line plots.
        See: https://stackoverflow.com/a/14860958/11362192"""
        return (self.MARKERSIZE_LIMITS[0]**2, self.MARKERSIZE_LIMITS[1]**2)

    @property
    def PALETTE(self) -> Tuple[str, ...]:
        """Get the current color palette as tuple."""
        return tuple(COLOR.PALETTE)
    
CATEGORY = _Category_()


@dataclass(frozen=True)
class _Distribution_:

    _ignore_: Tuple[str, ...] = ('levy_stable', 'studentized_range')

    COMMON: Tuple[str, ...] = (
        'norm', 'chi2', 'foldnorm', 'rayleigh', 'weibull_min', 'gamma', 'wald',
        'expon', 'logistic', 'lognorm')
    """Get distributions that commonly occur in practice."""

    @property
    def POSSIBLE(self) -> Tuple[str, ...]:
        """Get all possible continuous distributions from Scipy, except 
        'levy_stable' and 'studentized_range'."""
        return tuple(d for d in _distn_names if d not in self._ignore_)

    @property
    def COMMON_NOT_NORM(self) -> Tuple[str, ...]:
        """Get all common distributions but norm, usefull to fit if norm
        tests fail."""
        return tuple(d for d in self.COMMON if d != 'norm')
    
DIST = _Distribution_()


@dataclass(frozen=True)
class _Default_:

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
    QUANTILE_RANGES: Tuple[float, float, float] = (0.682, 0.954, 0.997)
    """Default quantile ranges for the Quantile Plotter.

    These values represent the quantiles corresponding to the standard 
    deviations (σ) of a normal distribution:
    - 0.682 = approximately ±1 σ, capturing about 68.2% of the data.
    - 0.954 = approximately ±2 σ, capturing about 95.4% of the data.
    - 0.997 = approximately ±3 σ, capturing about 99.7% of the data.
    """
    AGREEMENTS: Tuple[int, int, int] = (2, 4, 6)
    """Default agreement levels for the Quantile Plotter and for the
    quantiles in GaussianKDE and Violine Plotter.
    
    These values represent the quantiles corresponding to the standard
    deviations (σ) of a normal distribution:
    - 2 = ±1 σ, capturing about 68.2% of the data.
    - 4 = ±2 σ, capturing about 95.4% of the data.
    - 6 = ±3 σ, capturing about 99.7% of the data.
    """

    @property
    def MARKER(self) -> str:
        """Default marker for scatter plots."""
        marker = plt.rcParams['lines.marker']
        if marker == 'None':
            marker = CATEGORY.MARKERS[0]
        return marker
    
DEFAULT = _Default_()


@dataclass(frozen=True)
class _Anova_:

    SEP: Literal[':'] = ':'
    """Column name separator for interactions """
    INTERCEPT: str = INTERCEPT.name()
    """Column name for intercept"""
    EFFECTS: Literal['effects'] = 'effects'
    """"Default name for effects"""
    FEATURES: Literal['features'] = 'features'
    """"Default name for features"""
    SOURCE: Literal['Source'] = 'Source'
    """"Default name for source used in ANOVA table."""
    VIF: Literal['VIF'] = 'VIF'
    """"Default column name for VIF values in ANOVA table."""
    SMALLEST_INTERACTION: int = 2
    """Smallest possible interaction"""
    RESIDUAL: Literal['Residual'] = 'Residual'
    """Name in anova table for residual (not explained) values."""
    OBSERVATION: Literal['Observation'] = 'Observation'
    """Name in residual table for observation order."""
    PREDICTION: Literal['Prediction'] = 'Prediction'
    """Name in residual table for predicted values."""
    TOTAL: Literal['Total'] = 'Total'
    """Name in anova table for total (sum of the others) values."""
    PART: Literal['Part'] = 'Part'
    """Name in anova table for part values."""
    REPEATABILITY: Literal['EV'] = 'EV'
    """Name in anova table for repeatability values."""
    REPRODUCIBILITY: Literal['AV'] = 'AV'
    """Name in anova table for reproducibility values."""
    INTERACTION: Literal['Interaction'] = 'Interaction'
    """Name in anova table for interaction values."""
    RNR: Literal['R&R'] = 'R&R'
    """Name in rnr table for sum of R&R."""
    RESOLUTION_TO_UNCERTAINTY: float = 1 / 12**0.5
    """Factor for calculating the measurement uncertainty component for 
    the resolution. The uncertainty is assumed to be a rectangular 
    (uniformly distributed) distribution. The standard deviation of a 
    rectangular distribution is calculated using this factor. see:
    - https://de.wikipedia.org/wiki/Stetige_Gleichverteilung
    - https://www.versuchsmethoden.de/Mess-System-Analyse.pdf"""

    @property
    def TABLE_COLNAMES(self) -> List[str]:
        """Column names when crating the anova table using LinearModel 
        class"""
        return ['DF', 'SS', 'MS', 'F', 'p', 'n2']
    
    @property
    def VIF_COLNAMES(self) -> List[str]:
        """Column names when crating the vif table using the 
        `variance_inflation_factor` function."""
        return ['DF', self.VIF, 'GVIF', 'Threshold', 'Collinear', 'Method']
    
    @property
    def RNR_COLNAMES(self) -> List[str]:
        """Column names when crating the rnr table using the 
        `variance_inflation_factor` function."""
        return ['MS', 'MS/Total', 's', '6s', '6s/Total', '6s/Tolerance']

    @property
    def UNCERTAINTY_COLNAMES(self) -> List[str]:
        """Column names when crating the uncertainty table using the 
        `uncertainties` method of GageRnRModel."""
        return ['u', 'U', 'Q']

    @property
    def UNCERTAINTY_ROWS(self) -> List[str]:
        """Captions for the uncertainty table."""
        return ['RE', 'Bi', 'EVR', 'MS', 'EVO', 'AV', 'IA', 'MP']

ANOVA = _Anova_()


__all__ = [
    'SIGMA_DIFFERENCE',
    'LINE',
    'KW',
    'RE',
    'DIST',
    'COLOR',
    'LABEL',
    'ANOVA',
    'PLOTTER',
    'DEFAULT',
    'CATEGORY',
]