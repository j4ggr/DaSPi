# Plotting Guide

## Facets

The facet classes are used to place axes, labels and stripes (within the axes)  on the figure at the correct location and dimensions.

### AxesFacets

This class creates the layout for the subplots (axes) and the corresponding figure object. Initialization is heavily based on Matplotlib's `plt.subplots()` function (see [Matplotlib API](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)). This function is also called under the hood.

The layout of the subplots can be determined using the `nrows`, `ncols`, `width_ratios`, and `height_ratios` arguments, or the `mosaic` argument. You can find out more about the mosaic layout [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot_mosaic.html).

Here is an example of how to use the `AxesFacets` class:

``` py
import daspi as dsp

axes = dsp.AxesFacets(
    nrows=2, ncols=2, width_ratios=[3, 1], height_ratios=[1, 3])
```

And the counterpart using the mosaic argument:

``` py
axes = dsp.AxesFacets(mosaic=[
    'aaa.',
    'bbbc',
    'bbbc',
    'bbbc'])
```

For both, you'll get the following layout, with a difference being the subplot in the top right. In the mosaic, we have a '.' there. Matplotlib interprets this as empty space and does not create an Axes object at this location. This option isn't available with the first variant. Another difference is that in the mosaic notation above, the underlying matrix has the same size as the mosaic matrix. Therefore, we use row and column spans instead of width and height ratios. This means that in this example, the Axes object is present in 9 positions in the matrix. For this reason, it is also recommended to combine Mosaic with ratios:

``` py
axes = dsp.AxesFacets(
    mosaic=['a.', 'bc'], width_ratios=[3, 1], height_ratios=[1, 3])
```

![Mosaic Layout](../img/facets_axes-mosaic.png)

The individual axes can be retrieved using the getitem notation. Either a single number can be specified, which returns the unique Axes object in a flat list (from top left to bottom right) at the corresponding index. Alternatively, a tuple can also be specified, just like with numpy arrays. Using the example layout above, if you want the large Axes object, you have the following two options. Note that there is no Axes object in the top right corner.

``` py
axes[1]
axes[-1, 0]
```

The AxesFacets object also serves as an iterator. Like indexing, iterates over the Axes objects from top left to bottom right.

### StripesFacets

This class is used to create lines and areas (horizontal or vertical) that are placed within the subplots. The lines are used to visualize specification limits, control limits or the global mean or median. The areas are used to visualize the confidence interval of a line. The stripes are added to the Axes object as matplotlib.Line2D and matplotlib.Patch objects. These stripes are always straight lines that run parallel to an axis.

Imagine you're plotting data in some way on a series of subplots. Each subplot contains the same representation with similar data, but from a different category. Now you want to know if the mean values ​​of these split categories differ. This is where these stripes come into play.
Here's an example using the dataset with the dissolution time of aspirin. On the x-axis, we have the temperature, on the y-axis the dissolution time, and between the subplots, we divide by employee: **Important**, set `sharey` to `True`!

``` py
import daspi as dsp
import matplotlib.pyplot as plt

df = dsp.load_dataset('aspirin-dissolution')

fig, axes = plt.subplots(
    nrows=1, ncols=df['employee'].nunique(), sharex=True, sharey=True)

for ax, (name, group) in zip(axes, df.groupby('employee')):
    ax.scatter(group['temperature'], group['dissolution'])
    ax.set_title(str(name))
```

![Stripes](../img/facets_stripes-missing.png)

As we can see, it's difficult to say whether the mean value changes between the individual subplots. Now we want to know if the dissolution times differ between the employees and we also plot the upper specification limit of 25 s to show which employee's tablets took too long to dissolve. To achieve this, we create a `StripesFacets` object within the for loop and initialize it with the target data.

``` py
import daspi as dsp
import matplotlib.pyplot as plt

df = dsp.load_dataset('aspirin-dissolution')

fig, axes = plt.subplots(
    nrows=1, ncols=df['employee'].nunique(), sharex=True, sharey=True)

for ax, (name, group) in zip(axes, df.groupby('employee')):
    stripes = dsp.StripesFacets(
        group['dissolution'],
        target_on_y=True,
        single_axes=False,
        mean=True,
        confidence=0.95,
        spec_limits=dsp.SpecLimits(upper=25))
    ax.scatter(group['temperature'], group['dissolution'])
    ax.set_title(str(name))
    stripes.draw(ax)
```

![Stripes](../img/facets_stripes-drawn.png)

### LabelFacets

With this class you can add figure titles, subtitles, axis labels, column and row labels, a figure legend outside the axes or an info text at the bottom left of the diagram.

These facets are added to the Figure object as matplotlib.Text objects. The subplot area is automatically resized to prevent overlap.

``` py
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

axes = dsp.AxesFacets(nrows=3, ncols=2, sharey=True)

legend_data={
    'Lines': [
        (Line2D([0], [0], c='r'), Line2D([0], [0], c='b')),
        ('red line', 'blue line')],
    'Patches': [
        (Patch(color='r'), Patch(color='b')), 
        ('red patch', 'blue patch')]},

labels = dsp.LabelFacets(
    axes,
    fig_title='Title',
    sub_title='Subtitle',
    xlabel=('xlabel tl', 'xlabel tr', 'xlabel cl', 'xlabel cr', 'xlabel bl', 'xlabel br'),
    ylabel='single ylabel at center',
    info='Info goes here',
    cols=('col 1', 'col 2'),
    col_title='Column title',
    rows=('row 1', 'row 2', 'row3'),
    row_title='Row title',
    legend_data=legend_data)
labels.draw()
```

![Label Facets](../img/facets_labels.png)

### Combine facets

As a brief review, we combine these three classes AxesFacets, StripesFacets and LabelFacets in one figure using the example of aspirin dissolution time.

``` py
import daspi as dsp

df = dsp.load_dataset('aspirin-dissolution')

# Create the subplots layout
axes = dsp.AxesFacets(
    nrows=1, ncols=df['employee'].nunique(), sharex=True, sharey=True)

# Draw the stripes and plot data
for ax, (name, group) in zip(axes, df.groupby('employee')):
    stripes = dsp.StripesFacets(
        group['dissolution'],
        target_on_y=True,
        single_axes=False,
        mean=True,
        confidence=0.95,
        spec_limits=dsp.SpecLimits(upper=25))
    ax.scatter(group['temperature'], group['dissolution'])
    stripes.draw(ax)

# Label to create clarity
legend_data = {'Lines': stripes.handles_labels()}

labels = dsp.LabelFacets(
    axes,
    fig_title='Aspirin Dissolution Analysis',
    sub_title='Dissolution time ~ temperature + employee',
    xlabel='Temperature (°C)',
    ylabel='Dissolution time (s)',
    info='Mini-project from the Six Sigma Black Belt training',
    cols=tuple(df['employee'].unique()),
    col_title='Employee',
    legend_data=legend_data)
labels.draw()
```

![Aspirin Dissolution](../img/facets_combined.png)

## Plotters

The plotting library within DaSPi offers a wide selection of different plotters. Here's an overview of the different types and what they look like in use.

### Bivariate (XY) Plots

This is the most common type of plot, and it is used to visualize the relationship between two variables. The plotter uses the following parameters:
- `source`: The source data frame.
- `target`: The target variable (Y-axis).
- `feature`: The feature variable (X-axis).

The available plotters are:

![XY Plotters](../img/plotters_xy.png)

### Univariate (distribution) Plots

These types of plots are used to show the location and/or spread of the data points. The target variable is always continuous and the features are either absent or categorical.

![Univariate Plotters](../img/plotters_univariate.png)

### Plots for differences

These types of plots are used to show the difference between two or more variables. The target variable is always continuous and the features must be categorical.

![Difference Plotters](../img/plotters_differences.png)

### Special Plots

Here are some application-specific but still helpful plots.

![Special Plotters](../img/plotters_special.png)

Here an example of a loess line plot in combination with a scatter plot:

``` py
import daspi as dsp

df = dsp.load_dataset('iris')
axes = dsp.AxesFacets(nrows=1, ncols=1)
kwds = dict(
    source=df,
    target='length',
    feature='width',
    color=dsp.DEFAULT.PLOTTING_COLOR,
    ax=axes[0])

loess_plot = dsp.LoessLine(show_ci=True, **kwds)
loess_plot()
scatter_plot = dsp.Scatter(**kwds)
scatter_plot()
```

![XY Plot](../img/plotters_xy-example.png)