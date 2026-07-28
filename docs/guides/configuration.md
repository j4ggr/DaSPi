# Configuration Guide

DaSPi provides a centralized configuration system through the `CONFIG` object for managing global settings like language, username, and plotting styles.

---

## Quick Start

Access configuration through the `CONFIG` object:

```python
import daspi as dsp

# Set individual properties
dsp.CONFIG.language = 'de'
dsp.CONFIG.username = 'analyst'
dsp.CONFIG.style = 'ggplot'
```

---

## Configuration Properties

### Language

Controls the language for localized strings in charts and reports.

**Supported languages:**
- `'en'` — English (default)
- `'de'` — German
- `'fr'` — French

```python
# Set language
dsp.CONFIG.language = 'de'

# Now strings are in German
print(dsp.STR.accepted)  # Output: 'akzeptiert'
```

### Username

Sets the username displayed in chart annotations and info text.

```python
# Set username
dsp.CONFIG.username = 'j4ggr'

# Username appears in chart info text
chart = dsp.SingleChart(
    source=df,
    target='measurement',
    feature='sample_id'
).plot(dsp.Scatter).label(info=True)
```

By default, the username is read from the `USERNAME` environment variable.

### Style

Controls the matplotlib plotting style for all visualizations.

```python
# Set plotting style
dsp.CONFIG.style = 'ggplot'

# All subsequent plots use ggplot style
chart = dsp.SingleChart(...).plot(...)
```

**Available styles:**
- `'daspi'` — DaSPi default style (recommended)
- `'ggplot'` — ggplot2-inspired style
- `'seaborn-v0_8'` — Seaborn style
- Any matplotlib style name

---

## Configure Multiple Settings

Use the `configure()` method to set multiple properties at once:

```python
dsp.CONFIG.configure(
    language='fr',
    username='analyst',
    style='seaborn-v0_8'
)
```

---

## Temporary Changes with Context Managers

Context managers allow temporary configuration changes that automatically revert when the context exits.

### Temporary Language Change

```python
# Default language is English
print(dsp.STR.accepted)  # 'accepted'

# Temporarily switch to German
with dsp.CONFIG.use_language('de'):
    print(dsp.STR.accepted)  # 'akzeptiert'
    # Generate German report here

# Automatically reverts to English
print(dsp.STR.accepted)  # 'accepted'
```

### Temporary Style Change

```python
# Default DaSPi style
chart1 = dsp.SingleChart(...).plot(...)

# Temporarily use seaborn style
with dsp.CONFIG.use_style('seaborn-v0_8'):
    chart2 = dsp.SingleChart(...).plot(...)
    # This chart uses seaborn style

# Back to DaSPi style
chart3 = dsp.SingleChart(...).plot(...)
```

### Nested Context Managers

Context managers can be nested for complex workflows:

```python
# Generate reports in multiple languages
for lang in ['en', 'de', 'fr']:
    with dsp.CONFIG.use_language(lang):
        # Generate report in current language
        model = dsp.LinearModel(...)
        dsp.ResidualsCharts(model).plot().label(info=True)
```

---

## Reset to Defaults

Reset all configuration to default values:

```python
dsp.CONFIG.reset()
```

This sets:
- Language to `'en'`
- Username to environment variable or `'user'`
- Style to `'daspi'`

---

## Advanced Usage

### Multilingual Reports

Generate the same analysis in multiple languages:

```python
import daspi as dsp

df = dsp.load_dataset('painkillers-dissolution')
model = dsp.LinearModel(
    source=df,
    target='dissolution',
    factors=['employee', 'brand', 'catalyst']
)
model.recursive_elimination()

# Generate report in each language
for lang in ['en', 'de', 'fr']:
    with dsp.CONFIG.use_language(lang):
        chart = dsp.ResidualsCharts(model).plot().stripes().label(
            fig_title=f'Residuals Analysis ({lang.upper()})',
            info=True
        )
        chart.fig.savefig(f'residuals_{lang}.png', dpi=300)
```

### Style Comparison

Compare different plotting styles:

```python
styles = ['daspi', 'ggplot', 'seaborn-v0_8']

for style in styles:
    with dsp.CONFIG.use_style(style):
        chart = dsp.SingleChart(
            source=df,
            target='measurement',
            feature='sample'
        ).plot(dsp.Scatter).label(fig_title=f'Style: {style}')
        chart.fig.savefig(f'chart_{style}.png', dpi=300)
```

### User-Specific Settings

Set user preferences at the start of a notebook:

```python
import daspi as dsp

# User preferences
dsp.CONFIG.configure(
    language='de',
    username='reto.jaeggli',
    style='daspi'
)

# Now all analysis uses these settings
# ...
```

---

## See Also

- **[Plotting Guide](plotting.md)** — Visual customization and chart styling
- **[Localization (STR object)](../api/strings.md)** — Direct access to localized strings
