import sys
import pytest
import warnings
import matplotlib.pyplot as plt

from pathlib import Path
from pandas.core.frame import DataFrame

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi.plotlib.appearance import Style


class TestStyle:
    """Tests for the Style class in daspi.plotlib.appearance."""

    def test_available_styles(self) -> None:
        """Test that available styles include all matplotlib styles and more."""
        style = Style()
        daspi_styles = style.available
        matplotlib_styles = plt.style.available
        assert isinstance(daspi_styles, tuple)
        assert len(daspi_styles) > 0
        assert len(daspi_styles) > len(matplotlib_styles)
        assert all(s in daspi_styles for s in matplotlib_styles)
        assert 'default' not in daspi_styles

    def test_use_available_style(self) -> None:
        """Test switching to every available style works and updates current."""
        style = Style()
        daspi_styles = style.available
        for s in daspi_styles:
            style.use(s)
            assert style.current == s
            # Should not raise, but context may not be None for custom styles
            ctx = plt.style.context(s)
            assert hasattr(ctx, '__enter__')

    def test_use_invalid_style(self) -> None:
        """Test using an invalid style raises ValueError with correct message."""
        style = Style()
        # Accept any error message containing 'not found' or 'available styles'
        with pytest.raises(ValueError, match=r'(not found|available styles)') as exc:
            style.use('invalid_style')
        assert 'invalid_style' in str(exc.value)

    def test_use_default_style(self) -> None:
        """Test switching to default style works."""
        style = Style()
        style.use('default')
        assert style.current == 'default'
        ctx = plt.style.context('default')
        assert hasattr(ctx, '__enter__')

    def test_use_empty_string(self) -> None:
        """Test using empty string raises ValueError."""
        style = Style()
        with pytest.raises(ValueError, match=r'(not found|available styles)'):
            style.use('')

    def test_use_repeated(self) -> None:
        """Test repeated use of the same style does not error and updates current."""
        style = Style()
        s = style.available[0]
        style.use(s)
        assert style.current == s
        style.use(s)
        assert style.current == s
        
    def test_save_and_load_style(self) -> None:
        """Test saving the current style to a file in a temp directory and loading it back, applying to a new Style instance."""
        import tempfile
        import os
        style1 = Style()
        s = style1.available[0]
        style1.use(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            style_file = style1.save(tmpdir, style1.current)
            style2 = Style()
            style2.use(style_file)
            assert style2.current == style_file.stem
            assert style2.current == style1.current
