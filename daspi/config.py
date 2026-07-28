"""Central configuration management for DaSPi.

This module provides a unified interface for managing global settings
such as language, username, plotting style, and other configuration
options. All settings support context managers for temporary changes.

The `Config` class exposes properties for language, username, and
plotting style, delegating to the appropriate underlying systems
(strings.STR for language/username, plotlib.appearance for styles).
Changes can be made directly via property assignment or in bulk via
the ``configure()`` method.

Context managers
----------------
Both ``use_language()`` and ``use_style()`` allow temporary setting
changes that automatically revert when the context exits, making them
ideal for generating reports or charts in multiple languages or with
different visual styles without permanently altering the global state.

Usage
-----
Access the global configuration instance via::

```python
from daspi import CONFIG

CONFIG.language = 'de'
CONFIG.username = 'analyst'
CONFIG.style = 'ggplot'

# Temporary style change
with CONFIG.use_style('seaborn-v0_8'):
    # Create plots with seaborn style
    pass
# Style automatically reverts
```

Global instance
---------------
A singleton ``CONFIG`` instance is created and exported from this
module. It is also available from the top-level ``daspi`` package.
"""
from contextlib import contextmanager
from typing import Any, Literal

from .strings import STR

__all__ = ['CONFIG', 'Config']


class Config:
    """Central configuration manager for DaSPi settings.
    
    This class provides a unified interface for managing global
    settings that affect DaSPi behavior, including the language for
    localized strings, the username displayed in chart annotations,
    and the matplotlib plotting style.
    
    All properties support both direct assignment and temporary
    changes via context managers. The language and username properties
    delegate to the global ``STR`` instance from :mod:`daspi.strings`,
    while the style property manages matplotlib styles via
    :mod:`daspi.plotlib.appearance`.
    """
    
    def __init__(self) -> None:
        self._style: str = 'daspi'
    
    # Language property (delegates to STR)
    @property
    def language(self) -> Literal['en', 'de', 'fr']:
        """Language for localized strings ('en', 'de', 'fr')."""
        return STR.language
    
    @language.setter
    def language(self, value: Literal['en', 'de', 'fr']) -> None:
        STR.language = value
    
    # Username property (delegates to STR)
    @property
    def username(self) -> str:
        """Username displayed in chart info text."""
        return STR.username
    
    @username.setter
    def username(self, value: str) -> None:
        STR.username = value
    
    # Style property
    @property
    def style(self) -> str:
        """Current matplotlib style name."""
        return self._style
    
    @style.setter
    def style(self, value: str) -> None:
        """Set matplotlib style and cache the name."""
        from .plotlib.appearance import style
        style.use(value)
        self._style = value
    
    # Context managers
    def use_style(self, style_name: str):
        """Context manager for temporary style change.
        
        Parameters
        ----------
        style_name : str
            Name of matplotlib style to use temporarily
        
        Yields
        ------
        None
            Context in which the temporary style is active
        
        Examples
        --------
        ```python
        with CONFIG.use_style('seaborn-v0_8'):
            chart.show()  # Uses seaborn style
        # Style reverts to previous value
        ```
        """
        @contextmanager
        def _context():
            old_style = self._style
            try:
                self.style = style_name
                yield
            finally:
                self.style = old_style
        
        return _context()
    
    def use_language(self, lang: Literal['en', 'de', 'fr']):
        """Context manager for temporary language change.
        
        Parameters
        ----------
        lang : Literal['en', 'de', 'fr']
            Language code to use temporarily
        
        Yields
        ------
        None
            Context in which the temporary language is active
        
        Examples
        --------
        ```python
        with CONFIG.use_language('de'):
            title = STR.accepted  # returns 'akzeptiert'
        # Language reverts to previous value
        ```
        """
        return STR.use_language(lang)
    
    def configure(self, **kwargs: Any) -> None:
        """Configure multiple settings at once.
        
        Parameters
        ----------
        **kwargs : Any
            Key-value pairs of settings to update. Valid keys are
            'language', 'username', and 'style'
        
        Raises
        ------
        AttributeError
            If an invalid setting name is provided
        
        Examples
        --------
        ```python
        CONFIG.configure(
            language='de',
            username='j4ggr',
            style='ggplot'
        )
        ```
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"Config has no setting '{key}'. "
                    f"Valid settings: language, username, style"
                )
            setattr(self, key, value)
    
    def reset(self) -> None:
        """Reset all settings to defaults.
        
        Sets language to 'en', username to the value from the
        USERNAME environment variable (or 'user' if not set), and
        style to 'daspi'.
        """
        from os import environ
        STR.language = 'en'
        STR.username = environ.get('USERNAME', 'user')
        self.style = 'daspi'

# Global configuration instance
CONFIG = Config()
