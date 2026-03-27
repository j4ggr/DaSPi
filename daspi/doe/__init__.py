"""Design of Experiments (DOE) package.

This package provides classes and utilities for constructing
experimental designs as run-order DataFrames. The designs are intended
to be used together with ``daspi.anova.LinearModel`` for response
analysis after the experiment has been carried out.

`build`
    Factor definition (``Factor``) and design builder classes:

    - ``FullFactorialDesignBuilder`` — full factorial for any number
      of levels per factor.
    - ``FullFactorial2kDesignBuilder`` — classical 2ᵏ full factorial
      with optional fold-over.
    - ``FractionalFactorialDesignBuilder`` — 2^(k−p) fractional
      factorial with user-defined or default generator strings.

All public names from the ``build`` module are re-exported at the
package level.
"""
from .build import *
