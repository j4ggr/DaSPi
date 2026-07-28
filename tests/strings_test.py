"""Tests for localized string management."""
import pytest

from daspi.strings import STR


class TestLocalizedStrings:
    """Test the _LocalizedString descriptor pattern."""
    
    def test_default_language_is_english(self) -> None:
        """Test that default language is English."""
        assert STR.language == 'en'
        assert STR.anderson_darling == 'Anderson-Darling'
    
    def test_access_string_english(self) -> None:
        """Test accessing localized strings in English."""
        STR.language = 'en'
        assert STR.ok == 'OK'
        assert STR.nok == 'NOK'
        assert STR.accepted == 'accepted'
        assert STR.rejected == 'rejected'
    
    def test_access_string_german(self) -> None:
        """Test accessing localized strings in German."""
        STR.language = 'de'
        assert STR.ok == 'IO'
        assert STR.nok == 'NIO'
        assert STR.accepted == 'akzeptiert'
        assert STR.rejected == 'abgelehnt'
    
    def test_access_string_french(self) -> None:
        """Test accessing localized strings in French."""
        STR.language = 'fr'
        assert STR.ok == 'OK'
        assert STR.nok == 'NOK'
        assert STR.accepted == 'accepté'
        assert STR.rejected == 'rejeté'
    
    def test_language_switch(self) -> None:
        """Test that switching language changes returned strings."""
        # Start with English
        STR.language = 'en'
        assert STR.lsl == 'LSL'
        
        # Switch to German
        STR.language = 'de'
        assert STR.lsl == 'USG'
        
        # Switch to French
        STR.language = 'fr'
        assert STR.lsl == 'LSL'
        
        # Back to English
        STR.language = 'en'
        assert STR.lsl == 'LSL'
    
    def test_getitem_method(self) -> None:
        """Test __getitem__ method works with descriptors."""
        STR.language = 'en'
        assert STR['ok'] == 'OK'
        assert STR['accepted'] == 'accepted'
        
        STR.language = 'de'
        assert STR['ok'] == 'IO'
        assert STR['accepted'] == 'akzeptiert'
    
    def test_getitem_invalid_attribute(self) -> None:
        """Test __getitem__ with invalid attribute returns empty string."""
        with pytest.warns(UserWarning, match='No string found for invalid_attr'):
            result = STR['invalid_attr']
        assert result == ''
    
    def test_all_strings_have_all_languages(self) -> None:
        """Test that all localized strings work in all languages."""
        string_attrs = [
            'anderson_darling', 'ok', 'nok', 'accepted', 'rejected', 'borderline',
            'lsl', 'usl', 'lcl', 'ucl', 'excess', 'skew', 'kde_ax_label', 'stripes',
            'ci', 'formula', 'effects_label', 'ss_label', 'data_range',
            'paramcharts_fig_title', 'paramcharts_sub_title', 'paramcharts_feature_label',
            'residcharts_fig_title', 'resid_name', 'fit', 'charts_flabel_quantiles',
            'charts_flabel_density', 'charts_flabel_predicted', 'charts_flabel_observed',
            'charts_label_alpha_th', 'cp', 'cpk', 'paircharts_fig_title',
            'paircharts_sub_title', 'gstudycharts_fig_title', 'gstudycharts_sub_title',
            'rnrcharts_fig_title', 'rnrcharts_sub_title', 'rnrcharts_spread_proportions',
            'rnrcharts_suitability', 'lm_table_caption_summary', 'lm_table_caption_statistics',
            'lm_table_caption_anova', 'lm_table_caption_vif', 'lm_table_caption_rnr',
            'lm_table_rnr_source', 'lm_table_caption_ref_gages', 
            'lm_table_caption_ms_uncertainty', 'lm_table_caption_mp_uncertainty',
            'lm_table_caption_capabilities'
        ]
        
        for lang in ['en', 'de', 'fr']:
            STR.language = lang
            for attr in string_attrs:
                value = getattr(STR, attr)
                assert isinstance(value, str), f"{attr} should return str for language {lang}"
                assert len(value) > 0, f"{attr} should not be empty for language {lang}"
    
    def test_invalid_language_raises_assertion(self) -> None:
        """Test that setting invalid language raises AssertionError."""
        with pytest.raises(AssertionError):
            STR.language = 'es'  # type: ignore
    
    def test_descriptor_returns_string_not_dict(self) -> None:
        """Test that accessing attributes returns string, not dict."""
        STR.language = 'en'
        value = STR.ok
        assert isinstance(value, str)
        assert value == 'OK'
        # Make sure it's not a dict
        assert not isinstance(value, dict)
    
    def test_context_manager_temporary_language(self) -> None:
        """Test context manager for temporary language change."""
        # Start with English
        STR.language = 'en'
        assert STR.accepted == 'accepted'
        
        # Use context manager to temporarily switch to German
        with STR.use_language('de'):
            assert STR.accepted == 'akzeptiert'
            assert STR.ok == 'IO'
        
        # Should automatically revert to English
        assert STR.accepted == 'accepted'
        assert STR.ok == 'OK'
    
    def test_context_manager_nested(self) -> None:
        """Test nested context managers."""
        STR.language = 'en'
        
        with STR.use_language('de'):
            assert STR.ok == 'IO'
            
            with STR.use_language('fr'):
                assert STR.ok == 'OK'
            
            # Back to German
            assert STR.ok == 'IO'
        
        # Back to English
        assert STR.ok == 'OK'
    
    def teardown_method(self) -> None:
        """Reset language to English after each test."""
        STR.language = 'en'
