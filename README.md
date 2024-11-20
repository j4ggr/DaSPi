# Data analysis, Statistics and Process improvements (DaSPi)

Visualize and analyze your data with DaSPi. This package is designed for users who want to find relevant influencing factors in processes and validate improvements.
This package offers many [Six Sigma](https://en.wikipedia.org/wiki/Six_Sigma) tools based on the following packages:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://scipy.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

The goal of this package is to be easy to use and flexible so that it can be adapted to a wide array of data analysis tasks.

## Why DaSPi?

There are great packages for data analysis and visualization in Python, such as [Pandas](https://pandas.pydata.org/pandas-docs/stable), [Seaborn](https://seaborn.pydata.org/index.html), [Altair](https://altair-viz.github.io/), [Statsmodels](https://www.statsmodels.org/stable/), [Scipy](https://docs.scipy.org/doc/scipy/), [Pinguins](https://pingouin-stats.org/). But most of the time they work not directly with each other. Wouldn't it be great if you could use all of these packages together in one place? That's where DaSPi comes in. DaSPi is a Python package that provides a unified interface for data analysis, statistics and visualization. It allows you to use all of the great packages mentioned above together in one place, making it easier to explore and understand your data.

## Features

- **Ease of Use:** DaSPi is designed to be easy to use, even for beginners. It provides a simple and intuitive interface that makes it easy to get started with data analysis.
- **Flexibility:** DaSPi is highly flexible and can be adapted to a wide range of data analysis tasks. It provides a wide range of features and functions that can be customized to meet your specific needs.
- **Visualization:** DaSPi provides a wide range of visualization options, including charts, graphs, and maps. This makes it easy to explore and understand your data in a visual way.
- **Statistics:** DaSPi provides a wide range of statistical functions and tests, including hypothesis testing, confidence intervals, and regression analysis. This makes it easy to explore and understand your data in a statistical way.
- **Open Source:** DaSPi is open source, which means that it is free to use and modify. This makes it a great option for users who want to customize the package to their specific needs.

This Package contains following submodules:

- **plotlib:** Visualizations with Matplotlib, where the division by color, marker size or shape as well as rows and columns subplots are automated depending on the given categorical data. Any plots can also be combined, such as scatter with contour plot, violin with error bars or other creative combinations.
- **anova:** analysis of variance (ANOVA), which is used to compare the variance within and between of two or more groups, or the effects of different treatments on a response variable. It also includes a function for calculating the variance inflation factor (VIF) for linear regression models. The main class is LinearModel, which provides methods for fitting linear regression with interactions and automatically elimiinating insignificant variables.
- **statistics:** applied statistics, hypothesis test and confidence calculations. It also includes estimation for process capability and capability index.
- **datasets:** data for exersices. It includes different datasets that can be used for testing and experimentation.

## Usage

To use DaSPi, you can import the package and start exploring your data. Here is an example of how to use DaSPi to analyze a dataset:

``` py
import daspi
df = daspi.load_dataset('iris')

chart = daspi.MultipleVariateChart(
        source=df,
        target='length',
        feature='width',
        hue='species',
        col='leaf',
        markers=('x',),
        stretch_figsize=False,
    ).plot(
        daspi.GaussianKDEContour
    ).plot(
        daspi.Scatter
    ).label(
        feature_label='leaf width (cm)',
        target_label='leaf length (cm)',
    )
```

![Iris sepal length species](docs/img/iris_sepal_length_species.png)
