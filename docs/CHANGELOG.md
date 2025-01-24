# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

__Types of changes__:

- _Added_ for new features.
- _Changed_ for changes in existing functionality.
- _Deprecated_ for soon-to-be removed features.
- _Removed_ for now removed features.
- _Fixed_ for any bug fixes.
- _Security_ in case of vulnerabilities.

## [1.1.1] - 2025-01-24

### Added

- Option target_on_y for MultivariateChart

## [1.1.0] - 2025-01-23

### Added

- The options rows and cols in label method of JointChart used to labell the axes rows and columns.
- The Estimation class got now the min and max properties.
- The option exclude added to Estimation describe method, used to exclude variables from the analysis.
- The describe method of ProcessEstimation returns now also ok and nok as percentage values.
- A Beeswarm plotter that is comparable to the Jitter plotter. However, with Beeswarm the points are not distributed randomly, but arranged from the inside to the outside.
- A Quantiles plotter that is comparable to a boxplot. However, quantiles does not draw whiskers but rather just several boxes that show the given number of percent of the observed data.
- The option to change color, marker, target_on_y, size and width in plot method for SingleChart and JointChart.
- The option hide_axis for each Plotter class during initialization.
- The option visible_spines for each Plotter class during initialization.

## Changed

- The class name MultipleVariateChart to MultivariateChart

## Removed

- The option show_density_axis for GaussianKDE. Use hide_axis and visible_spines instead.

## [1.0.1] - 2025-01-09

### Fixed

- Calculating Z-score for ProcessEstimator failure when no specification limit is provided.

## [1.0.0] - 2025-01-09

### Added

- ProcessEstimator z_transform method and tolerance_range property.
- GaussianKDE ignore_feature option for plotting all at same axis base or not.
- GaussianKDE fill option for filling in the curves.
- SpreadWidth center calculation option kind.
- ProcessEstimator Z_lt property for long-term sigma level.
- ProcessEstimator Z property as sigma level process capability.
- Confidence interval functions for Cp and Cpk in confidence module.
- The function estimate_capability_confidence function in estimation module.
- CapabilityConfidenceIntervall Plotter class.
- Datasets drop_card.csv and salt_sticks.csv.
- AxesFacets mosaic option for more flexability in axes arrangement.
- AxesFacets properties flat and shape.
- Loess class in estimation module used for plotting loess curves.
- Plotting styles ggplot2, daspi-dark and seaborn.
- The module appereance in plotlib used for managing loading and saving styles.
- Colormaps used at seaborn to the module appereance.

### Changed

- Estimator agreement property gets allways the multiplier of sigma agreement.
- Chart property name default_kwds changed to kw_default.
- JointChart possability to add axes independend stripes for mean, median and control_limits
- Errorbars requires now the amount of groups. This is necessary for bonferroni adjustment.
- Chart property name axes_facets changed to Axes.
- Most of the arguments is now only available as a keyword argument in Chart classes.
- When target_on_y is set to False in SingleChart, the x and y axes params are swapped.
- The default plotting color is now black and the palette behaves like ggplot2. That means that the colors changes with the number of groups (except for the first group).

### Removed

- The marker option for TransformPlotter.

### Fixed

- ErrorBarsPlotter and JitterPlotter now use the marker option.
- GaussianKDEPlotter color option now has an effect.
- JointChart handles and labels now get the stripes stuff from all axes and not only from the last one.
- Categorical floating features are now plotted as categorical.


## [0.5.1] - 2024-11-19

### Added

- GaussianKDEContour Plotter for plotting bivariate distributions as contour lines.
- kernel_density_estimation_2d function used to calculate bivariate kernel density estimations.
- Stripe, StripeLine and StripeSpan Plotter for plotting stripes using StripesFacets class or within BlandAltman Plotter.
- Documentation site

### Changed

- StripesFacets now uses the new StripeLine and StripeSpan Plotters.
- BlandAltmanPlotter now uses the new StripeLine and StripeSpan Plotters.
- The target_on_y option is moved from StripesFacets draw method to init.
- Getting Figure and Axes behavior if not given during initialization of a Plotter class. The plt.gca() function is now used.

### Removed

- Option marker when initializing a Line Plotter. Set the marker if needed when calling the plotter.

### Fixed

- Charts got wrong colors if not given during initialization.

## [0.4.0] - 2024-10-24

### Added

- BivariateUnivariateCharts template, provides a set of charts for visualizing the relationship between a target variable and a feature variable.
- Option for Plotters to change the marker. This can be set during initialization of the Plotter.
- Option for Charts to change markers, colors and amount of size bins for categorical differentiation.
- Single feature or target label for JointCharts if all aubplots shares the corressponding axis.

### Fixed

- Categorical feature failed when using floats. Categorical features failed when using floats. Dodging is now always performed, but only with one category if dodge was set to False.

## [0.3.0] - 2024-10-09

### Added

- PairComparisonCharts template, provides a set of charts for visualizing the pairwise comparison of two variables.
- kw_where argument to filter data when calling the plot method of a Chart object. For more information see the documentation [pandas.DataFrame.where](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.where.html)

### Removed

- BaseTemplate, all template chart classes inherits directly from JointChart.

## [0.2.0] - 2024-07-10

### Added

- Option for LinearModel and variance_inflation_factor function to choose if generalized or only straightforward VIFs should be calculated.
- Option for LinearModel if all categorical features should be one-hot encoded or not.
- ParameterRelevanceCharts template and BaseTemplate as parent for ResidualsCharts and ParameterRelevanceCharts.
- anguage dependent strings for sum of squares label, effects label and ParameterRelevanceCharts texts.
- Option to skip plotting nan values for all TransformPlotters.
- Property effect_threshold for LinearModel, which returns the effect value for the given alpha risk.
- Option to skip evaluation of intercept as least significant term for LinearModel.
- Property design_matrix for LinearModel, which returns the currently encoded design matrix.
- Option to draw a percentage line when using the Pareto plotter or not.
- GVIF, its threshold, collinearity decision and method used for the VIF table.

### Changed

- Variance inflation factor function calculates now only values for equal orders.
- Variance inflation factor function returns a DataFrame instead of a Series
- Renamed attribute of LinearModel: exclude -> excluded.
- least_term method of LinearModel when evaluating by effects and some effects are equal, the highest interaction order is returned. Now it also takes into account nan values.
- eliminate method of LinearModel returns self for method chaining.
- Effects are now calculated as absolute t-values.
- Effects table keeps now the original term order (was messed up).

### Fixed

- has_insignificant_term method of LinearModel could not detect minimal terms when the attribute skip_intercept_as_least was set to True.
- Degrees of freedom for VIF table was calculated wrong.

## [0.1.0] - 2024-07-01

Initial release.

### Added

- Character strings in different languages ​​for charts and tables. The language can be set initially and the output adapts automatically. Available languages: German and English.
- Constants that are available globally throughout the package.

__Anova module:__

- LinearModel for running regressions with higher order interactions and analyzing significant terms and their elimination.
- Tables such as ANOVA, VIF and residuals.

__Statistics module:__

- Estimator for calculating statistics such as mean, stdev, skew and excess.
- ProcessEstimator as an extension of the Estimator class which also takes specification limits into account to calculate Cp and Cpk values.
- Confidence interval functions for position, spread, proportions and fit lines.
- Estimation functions for kernel density and distributions
- Hypothesis tests for normality, homogeneity of variance and independence.

__Plotlib module:__

- SingleChart for plotting single Axes charts with different plotters.
- MultivariateChart for plotting charts with multiple Axes, with the plots divided into the individual Axes depending on the categorical variates.
- JointChart for plotting multiple axes to which different plotters can be applied.
- Any type of plotters such as Bar, Violin, KDE, Scatter, Line, Errorbar, Confidence Intervals, Pareto, Bland Altmann, Parallel Coordinate and Jitter.
- Pre-built Matplotlib stylesheet
- Classifiers to classify the plots into different colors, markers or marker sizes depending on the categorical variates.
- Facets for plotting multiple charts in one figure, with the plots divided into the individual Axes depending on the categorical variates. Also for adding stribes to the charts or adding labels, titles and legends.

