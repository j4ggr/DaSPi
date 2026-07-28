import warnings
from datetime import date
from os import environ
from typing import Literal

__all__ = ['STR']


class _LocalizedString:
    """Descriptor for localized strings that returns the translation
    for the current language."""
    
    def __init__(self, translations: dict[str, str]) -> None:
        self.translations = translations
    
    def __get__(self, obj: '_String_', objtype: type | None = None) -> str:
        if obj is None:
            return self  # type: ignore
        return self.translations[obj.language]


class _String_:

    anderson_darling = _LocalizedString({
        'en': 'Anderson-Darling',
        'de': 'Anderson-Darling',
        'fr': 'Anderson-Darling'})

    ok = _LocalizedString({ 
        'en': 'OK',
        'de': 'IO',
        'fr': 'OK'})

    nok = _LocalizedString({ 
        'en': 'NOK',
        'de': 'NIO',
        'fr': 'NOK'})
    
    accepted = _LocalizedString({
        'en': 'accepted',
        'de': 'akzeptiert',
        'fr': 'accepté'})
    
    rejected = _LocalizedString({
        'en': 'rejected',
        'de': 'abgelehnt',
        'fr': 'rejeté'})
    
    borderline = _LocalizedString({
        'en': 'borderline',
        'de': 'grenzwertig',
        'fr': 'limite'})

    lsl = _LocalizedString({ 
        'en': 'LSL',
        'de': 'USG',
        'fr': 'LSL'})
    
    usl = _LocalizedString({
        'en': 'USL',
        'de': 'OSG',
        'fr': 'USL'})
    
    lcl = _LocalizedString({ 
        'en': 'LCL',
        'de': 'UEG',
        'fr': 'LCL'})
    
    ucl = _LocalizedString({
        'en': 'UCL',
        'de': 'OEG',
        'fr': 'UCL'})
    
    excess = _LocalizedString({
        'en': 'excess',
        'de': 'Exzess',
        'fr': 'excès'})
    
    skew = _LocalizedString({
        'en': 'skew',
        'de': 'Schiefe',
        'fr': 'asymétrie'})
    
    kde_ax_label = _LocalizedString({
        'en': 'Estimated kernel density',
        'de': 'Geschätzte Kerndichte',
        'fr': 'Densité de noyau estimée'})

    stripes = _LocalizedString({
        'en': 'Lines',
        'de': 'Linien',
        'fr': 'Lignes'})

    ci = _LocalizedString({
        'en': 'CI',
        'de': 'KI',
        'fr': 'IC'})
    
    formula = _LocalizedString({
        'en': 'formula',
        'de': 'Formel',
        'fr': 'formule'})
    
    effects_label = _LocalizedString({
        'en': 'Standardized effect',
        'de': 'Standardisierter Effekt',
        'fr': 'Effet standardisé'})
    
    ss_label = _LocalizedString({
        'en': 'Sum of Squares',
        'de': 'Summenquadrate',
        'fr': 'Somme des carrés'})
    
    data_range = _LocalizedString({
        'en': 'Range',
        'de': 'Spannweite',
        'fr': 'Plage des données'})
    
    paramcharts_fig_title = _LocalizedString({
        'en': 'Parameter Analysis',
        'de': 'Parameter Analyse',
        'fr': 'Analyse des paramètres'})
    
    paramcharts_sub_title = _LocalizedString({
        'en': 'Relative importance of parameters',
        'de': 'Relative Wichtigkeit der Parameter',
        'fr': 'Importance relative des paramètres'})
    
    paramcharts_feature_label = _LocalizedString({
        'en': 'Parameter',
        'de': 'Parameter',
        'fr': 'Paramètre'})
    
    residcharts_fig_title = _LocalizedString({
        'en': 'Residuals analysis',
        'de': 'Residuen Analyse',
        'fr': 'Analyse des résidus'})
    
    resid_name = _LocalizedString({
        'en': 'Residuals',
        'de': 'Residuen',
        'fr': 'Résidus'})
    
    fit = _LocalizedString({
        'en': 'Fit',
        'de': 'Anpassung',
        'fr': 'Ajustement'})
    
    charts_flabel_quantiles = _LocalizedString({
        'en': 'Std. Normal Distribution quantiles',
        'de': 'Standardnormalverteilung Quantile',
        'fr': 'Quantiles de la distribution normale standard'})
    
    charts_flabel_density = _LocalizedString({
        'en': 'Estimated kernel density',
        'de': 'Geschätzte Kerndichte',
        'fr': 'Densité de noyau estimée'})
    
    charts_flabel_predicted = _LocalizedString({
        'en': 'Predicted values',
        'de': 'Vorhersage',
        'fr': 'Valeurs prédites'})
    
    charts_flabel_observed = _LocalizedString({
        'en': 'Observation order',
        'de': 'Beobachtungsreihenfolge',
        'fr': 'Ordre d\'observation'})
    
    charts_label_alpha_th = _LocalizedString({
        'en': r'effect_α\;(α={alpha})',
        'de': r'Effekt_α\;(α={alpha})',
        'fr': r'effet_α\;(α={alpha})'})

    cp = _LocalizedString({
        'en': 'Process Capability index Cp',
        'de': 'Prozessfähigkeitsindex Cp',
        'fr': 'Indice de capacité de processus Cp'})

    cpk = _LocalizedString({
        'en': 'Adjusted Process Capability index Cpk',
        'de': 'Angepasster Prozessfähigkeitsindex Cpk',
        'fr': 'Indice de capacité de processus ajusté Cpk'})
    
    paircharts_fig_title = _LocalizedString({
        'en': 'Pairwise analysis',
        'de': 'Paarweise Analyse',
        'fr': 'Analyse pair à pair'})
    
    paircharts_sub_title = _LocalizedString({
        'en': 'Bland-Altman 95 % CI and individual value comparison',
        'de': 'Bland-Altman 95 %-KI und Einzelwertvergleich',
        'fr': 'Bland-Altman 95 % IC et comparaison des valeurs individuelles'})
    
    gstudycharts_fig_title = _LocalizedString({
        'en': 'Measurement system analysis',
        'de': 'Messsystemanalyse',
        'fr': 'Analyse du système de mesure'})
    
    gstudycharts_sub_title = _LocalizedString({
        'en': 'Gage study type 1',
        'de': 'MSA Typ 1',
        'fr': 'Etude de système de mesure type 1'})
    
    rnrcharts_fig_title = _LocalizedString({
        'en': 'Measurement system analysis',
        'de': 'Messsystemanalyse',
        'fr': 'Analyse du système de mesure'})

    rnrcharts_sub_title = _LocalizedString({
        'en': 'Repeatability and reproducibility (Gage R&R)',
        'de': 'Wiederholbarkeit und Reproduzierbarkeit (Gage R&R)',
        'fr': 'Répétabilité et reproductibilité (Gage R&R)'})
    
    rnrcharts_spread_proportions = _LocalizedString({
        'en': 'Spread proportions',
        'de': 'Streuungsanteile',
        'fr': 'Proportions de dispersion'})
    
    rnrcharts_suitability = _LocalizedString({
        'en': 'Suitability index Q',
        'de': 'Eignungskennwert Q',
        'fr': 'Indice d\'adéquation Q'})
    
    lm_table_caption_summary = _LocalizedString({
        'en': 'Model summary',
        'de': 'Modellzusammenfassung',
        'fr': 'Résumé du modèle'})
    
    lm_table_caption_statistics = _LocalizedString({
        'en': 'Parameter statistics',
        'de': 'Parameterstatistik',
        'fr': 'Statistiques des paramètres'})
    
    lm_table_caption_anova = _LocalizedString({
        'en': 'Analysis of variance',
        'de': 'Varianzanalyse',
        'fr': 'Analyse de la variance'})
    
    lm_table_caption_vif = _LocalizedString({
        'en': 'Variance inflation factor',
        'de': 'Varianzinflationfaktor',
        'fr': 'Facteur d\'inflation de la variance'})
    
    lm_table_caption_rnr = _LocalizedString({
        'en': 'Repeatability and reproducibility (R&R)',
        'de': 'Wiederholbarkeit und Reproduzierbarkeit (R&R)',
        'fr': 'Répétabilité et reproductibilité (R&R)'})
    
    lm_table_rnr_source = _LocalizedString({
        'en': 'Source',
        'de': 'Quelle',
        'fr': 'Source'})
    
    lm_table_caption_ref_gages = _LocalizedString({
        'en': 'Reference analysis',
        'de': 'Analyse der Referenzen',
        'fr': 'Analyse des références'})

    lm_table_caption_ms_uncertainty = _LocalizedString({
        'en': 'Measurement uncertainty budget measuring system',
        'de': 'Messunsicherheitsbudget Messsystem',
        'fr': 'Système de mesure du budget d\'incertitude de mesure'})
    
    lm_table_caption_mp_uncertainty = _LocalizedString({
        'en': 'Measurement uncertainty budget measuring process',
        'de': 'Messunsicherheitsbudget Messprozess',
        'fr': 'Budget d\'incertitude de mesure de la méthode de mesure'})
    
    lm_table_caption_capabilities = _LocalizedString({
        'en': 'Gage capability',
        'de': 'Messsystemfähigkeit',
        'fr': 'Capacité du système de mesure'})

    _language_: Literal['en', 'de', 'fr'] = 'en'
    _username_: str = environ.get('USERNAME', 'user')

    @property
    def today(self) -> str:
        """Current date in YYYY-MM-DD format."""
        return date.today().strftime('%Y-%m-%d')
    
    @property
    def language(self) -> Literal['en', 'de', 'fr']:
        """Language (abbreviation) in which the strings should be
        rendered."""
        return self._language_
    
    @language.setter
    def language(self, lang: Literal['en', 'de', 'fr']) -> None:
        assert lang in ('en', 'de', 'fr'), f"Language must be 'en', 'de', or 'fr', got '{lang}'"
        self._language_ = lang
    
    @property
    def username(self) -> str:
        """Username reflected in the charts in the info text, defaults 
        to username from the environment variable."""
        return self._username_
    
    @username.setter
    def username(self, name: str) -> None:
        self._username_ = name
    
    def use_language(self, lang: Literal['en', 'de', 'fr']):
        """Context manager for temporary language change.
        
        Usage:
            with STR.use_language('de'):
                title = STR.accepted  # returns 'akzeptiert'
            # Language automatically reverts to previous value
        """
        from contextlib import contextmanager
        
        @contextmanager
        def _context():
            old_lang = self._language_
            try:
                self.language = lang
                yield
            finally:
                self._language_ = old_lang
        
        return _context()
    
    def __getitem__(self, item: str) -> str | Literal['']:
        _string = ''
        try:
            _string = getattr(self, item)  # Descriptor returns localized string
        except AttributeError:
            warnings.warn(f'No string found for {item}!')
        return _string
STR = _String_()
