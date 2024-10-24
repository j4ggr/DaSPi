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

## [0.4.0] - 2024-10-24

### Added

- BivariateUnivariateCharts template, provides  a set of charts for visualizing the relationship between a target variable and a feature variable.
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
- Constants that are available globally throughout the package

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
- MultipleVariateChart for plotting charts with multiple Axes, with the plots divided into the individual Axes depending on the categorical variates.
- JointChart for plotting multiple axes to which different plotters can be applied.
- Any type of plotters such as Bar, Violin, KDE, Scatter, Line, Errorbar, Confidence Intervals, Pareto, Bland Altmann, Parallel Coordinate and Jitter.
- Pre-built Matplotlib stylesheet
- Classifiers to classify the plots into different colors, markers or marker sizes depending on the categorical variates.
- Facets for plotting multiple charts in one figure, with the plots divided into the individual Axes depending on the categorical variates. Also for adding stribes to the charts or adding labels, titles and legends.

