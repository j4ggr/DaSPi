"""Tests for central configuration management."""
import pytest

from daspi import CONFIG, STR


class TestConfig:
    """Test the Config class for central settings management."""
    
    def test_default_values(self) -> None:
        """Test that default configuration values are set correctly."""
        # Reset to defaults first
        CONFIG.reset()
        assert CONFIG.language == 'en'
        assert CONFIG.style == 'daspi'
        # username depends on environment, just check it's a string
        assert isinstance(CONFIG.username, str)
    
    def test_language_property(self) -> None:
        """Test that language property delegates to STR."""
        CONFIG.language = 'de'
        assert CONFIG.language == 'de'
        assert STR.language == 'de'
        
        # Setting STR.language should also update CONFIG
        STR.language = 'fr'
        assert CONFIG.language == 'fr'
    
    def test_username_property(self) -> None:
        """Test that username property delegates to STR."""
        CONFIG.username = 'test_user'
        assert CONFIG.username == 'test_user'
        assert STR.username == 'test_user'
        
        # Setting STR.username should also update CONFIG
        STR.username = 'another_user'
        assert CONFIG.username == 'another_user'
    
    def test_style_property(self) -> None:
        """Test that style property works correctly."""
        # Set to a known style
        CONFIG.style = 'default'
        assert CONFIG.style == 'default'
        
        # Setting back to daspi
        CONFIG.style = 'daspi'
        assert CONFIG.style == 'daspi'
    
    def test_configure_single_setting(self) -> None:
        """Test configure() with single setting."""
        CONFIG.configure(language='de')
        assert CONFIG.language == 'de'
    
    def test_configure_multiple_settings(self) -> None:
        """Test configure() with multiple settings."""
        CONFIG.configure(
            language='fr',
            username='j4ggr',
            style='ggplot'
        )
        assert CONFIG.language == 'fr'
        assert CONFIG.username == 'j4ggr'
        assert CONFIG.style == 'ggplot'
    
    def test_configure_invalid_setting(self) -> None:
        """Test that configure() raises error for invalid setting."""
        with pytest.raises(AttributeError, match="Config has no setting 'invalid_key'"):
            CONFIG.configure(invalid_key='value')
    
    def test_use_language_context_manager(self) -> None:
        """Test use_language() context manager."""
        CONFIG.language = 'en'
        assert STR.ok == 'OK'
        
        with CONFIG.use_language('de'):
            assert STR.ok == 'IO'
            assert CONFIG.language == 'de'
        
        # Should revert
        assert STR.ok == 'OK'
        assert CONFIG.language == 'en'
    
    def test_use_language_nested(self) -> None:
        """Test nested use_language() context managers."""
        CONFIG.language = 'en'
        
        with CONFIG.use_language('de'):
            assert STR.accepted == 'akzeptiert'
            
            with CONFIG.use_language('fr'):
                assert STR.accepted == 'accepté'
            
            # Back to German
            assert STR.accepted == 'akzeptiert'
        
        # Back to English
        assert STR.accepted == 'accepted'
    
    def test_use_style_context_manager(self) -> None:
        """Test use_style() context manager."""
        CONFIG.style = 'daspi'
        assert CONFIG.style == 'daspi'
        
        with CONFIG.use_style('default'):
            assert CONFIG.style == 'default'
        
        # Should revert
        assert CONFIG.style == 'daspi'
    
    def test_use_style_nested(self) -> None:
        """Test nested use_style() context managers."""
        CONFIG.style = 'daspi'
        
        with CONFIG.use_style('default'):
            assert CONFIG.style == 'default'
            
            with CONFIG.use_style('ggplot'):
                assert CONFIG.style == 'ggplot'
            
            # Back to default
            assert CONFIG.style == 'default'
        
        # Back to daspi
        assert CONFIG.style == 'daspi'
    
    def test_reset(self) -> None:
        """Test reset() returns all settings to defaults."""
        # Change all settings
        CONFIG.configure(
            language='fr',
            username='custom_user',
            style='ggplot'
        )
        
        # Reset
        CONFIG.reset()
        
        # Check defaults
        assert CONFIG.language == 'en'
        assert CONFIG.style == 'daspi'
        # username resets to environment variable
        assert isinstance(CONFIG.username, str)
    
    def test_mixed_context_managers(self) -> None:
        """Test mixing different context managers."""
        CONFIG.language = 'en'
        CONFIG.style = 'daspi'
        
        with CONFIG.use_language('de'), CONFIG.use_style('default'):
            assert CONFIG.language == 'de'
            assert CONFIG.style == 'default'
        
        # Both should revert
        assert CONFIG.language == 'en'
        assert CONFIG.style == 'daspi'
    
    def test_style_sync_with_appearance_module(self) -> None:
        """Test that CONFIG.style reflects direct changes to plotlib.appearance.style.
        
        This test verifies the TODO fix: CONFIG.style should always return the
        current matplotlib style, even when changed directly via style.use().
        """
        from daspi.plotlib.appearance import style
        
        # Set initial style
        CONFIG.style = 'daspi'
        assert CONFIG.style == 'daspi'
        assert style.current == 'daspi'
        
        # Change style directly via appearance module (not through CONFIG)
        style.use('ggplot')
        
        # CONFIG.style should reflect this change
        assert CONFIG.style == 'ggplot', "CONFIG.style should reflect direct style.use() calls"
        assert style.current == 'ggplot'
        
        # Change back via appearance module
        style.use('default')
        assert CONFIG.style == 'default', "CONFIG.style should stay synchronized"
        assert style.current == 'default'
    
    def teardown_method(self) -> None:
        """Reset configuration after each test."""
        CONFIG.reset()
