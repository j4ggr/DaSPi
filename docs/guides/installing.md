
# Installation Guide

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## Installing from PyPI

The easiest way to install is via pip:

``` bash
pip install daspi
```

## Verifying Installation

Verify the installation by executing the following lines with a Python interpreter:

``` py
import daspi
print(daspi.__version__)
```

Or directly in the terminal:

``` bash
python -c "import daspi; print(daspi.__version__)" 
```

## Troubleshooting

### Common Issues

1. **Import Errors**
    - Ensure Python environment is activated
    - Verify package installation with `pip list`

2. **Version Conflicts**
    - Try creating a new virtual environment
    - Update dependencies with `pip install --upgrade`

### Getting Help

- Open an issue on [GitHub](https://github.com/j4ggr/DaSPi/issues)
- Check the documentation

## Next Steps

- Read the [Quick Start Guide](index.md)
- Check out the API References:
    - [Plotlib](../plotlib/chart/index.md)
    - [Statistics](../statistics/hypothesis/index.md)
    - [Anova](../anova/index.md)
- View [User Guide](../guides/index.md)
