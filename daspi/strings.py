import warnings

from os import environ
from typing import Dict
from typing import Literal
from datetime import date


class _String_:

    anderson_darling: Dict[str, str] = {
        'en': 'Anderson-Darling',
        'de': 'Anderson-Darling',
        'fr': 'Anderson-Darling'}

    ok: Dict[str, str] = { 
        'en': 'OK',
        'de': 'IO',
        'fr': 'OK'}

    nok: Dict[str, str] = { 
        'en': 'NOK',
        'de': 'NIO',
        'fr': 'NOK'}
    
    accepted: Dict[str, str] = {
        'en': 'accepted',
        'de': 'akzeptiert',
        'fr': 'accepté'}
    
    rejected: Dict[str, str] = {
        'en': 'rejected',
        'de': 'abgelehnt',
        'fr': 'rejeté'}
    
    borderline: Dict[str, str] = {
        'en': 'borderline',
        'de': 'grenzwertig',
        'fr': 'limite'}

    lsl: Dict[str, str] = { 
        'en': 'LSL',
        'de': 'USG',
        'fr': 'LSL'}
    
    usl: Dict[str, str] = {
        'en': 'USL',
        'de': 'OSG',
        'fr': 'USL'}
    
    lcl: Dict[str, str] = { 
        'en': 'LCL',
        'de': 'UEG',
        'fr': 'LCL'}
    
    ucl: Dict[str, str] = {
        'en': 'UCL',
        'de': 'OEG',
        'fr': 'UCL'}
    
    excess: Dict[str, str] = {
        'en': 'excess',
        'de': 'Exzess',
        'fr': 'excès'}
    
    skew: Dict[str, str] = {
        'en': 'skew',
        'de': 'Schiefe',
        'fr': 'asymétrie'}
    
    kde_ax_label: Dict[str, str] = {
        'en': 'Estimated kernel density',
        'de': 'Geschätzte Kerndichte',
        'fr': 'Densité de noyau estimée'}

    stripes: Dict[str, str] = {
        'en': 'Lines',
        'de': 'Linien',
        'fr': 'Lignes'}

    ci: Dict[str, str] = {
        'en': 'CI',
        'de': 'KI',
        'fr': 'IC'}
    
    formula: Dict[str, str] = {
        'en': 'formula',
        'de': 'Formel',
        'fr': 'formule'}
    
    effects_label: Dict[str, str] = {
        'en': 'Standardized effect',
        'de': 'Standardisierter Effekt',
        'fr': 'Effet standardisé'}
    
    ss_label: Dict[str, str] = {
        'en': 'Sum of Squares',
        'de': 'Summenquadrate',
        'fr': 'Somme des carrés'}
    
    data_range: Dict[str, str] = {
        'en': 'Range',
        'de': 'Spannweite',
        'fr': 'Plage des données'}
    
    paramcharts_fig_title: Dict[str, str] = {
        'en': 'Parameter Analysis',
        'de': 'Parameter Analyse',
        'fr': 'Analyse des paramètres'}
    
    paramcharts_sub_title: Dict[str, str] = {
        'en': 'Relative importance of parameters',
        'de': 'Relative Wichtigkeit der Parameter',
        'fr': 'Importance relative des paramètres'}
    
    paramcharts_feature_label: Dict[str, str] = {
        'en': 'Parameter',
        'de': 'Parameter',
        'fr': 'Paramètre'}
    
    residcharts_fig_title: Dict[str, str] = {
        'en': 'Residuals analysis',
        'de': 'Residuen Analyse',
        'fr': 'Analyse des résidus'}
    
    resid_name: Dict[str, str] = {
        'en': 'Residuals',
        'de': 'Residuen',
        'fr': 'Résidus'}
    
    fit: Dict[str, str] = {
        'en': 'Fit',
        'de': 'Anpassung',
        'fr': 'Ajustement'}
    
    charts_flabel_quantiles: Dict[str, str] = {
        'en': 'Std. Normal Distribution quantiles',
        'de': 'Standardnormalverteilung Quantile',
        'fr': 'Quantiles de la distribution normale standard'}
    
    charts_flabel_density: Dict[str, str] = {
        'en': 'Estimated kernel density',
        'de': 'Geschätzte Kerndichte',
        'fr': 'Densité de noyau estimée'}
    
    charts_flabel_predicted: Dict[str, str] = {
        'en': 'Predicted values',
        'de': 'Vorhersage',
        'fr': 'Valeurs prédites'}
    
    charts_flabel_observed: Dict[str, str] = {
        'en': 'Observation order',
        'de': 'Beobachtungsreihenfolge',
        'fr': 'Ordre d\'observation'}
    
    charts_label_alpha_th: Dict[str, str] = {
        'en': r'effect_α\;(α={alpha})',
        'de': r'Effekt_α\;(α={alpha})',
        'fr': r'effet_α\;(α={alpha})'}

    cp: Dict[str, str] = {
        'en': 'Process Capability index Cp',
        'de': 'Prozessfähigkeitsindex Cp',
        'fr': 'Indice de capacité de processus Cp'}

    cpk: Dict[str, str] = {
        'en': 'Adjusted Process Capability index Cpk',
        'de': 'Angepasster Prozessfähigkeitsindex Cpk',
        'fr': 'Indice de capacité de processus ajusté Cpk'}
    
    paircharts_fig_title: Dict[str, str] = {
        'en': 'Pairwise analysis',
        'de': 'Paarweise Analyse',
        'fr': 'Analyse pair à pair'}
    
    paircharts_sub_title: Dict[str, str] = {
        'en': 'Bland-Altman 95 % CI and individual value comparison',
        'de': 'Bland-Altman 95 %-KI und Einzelwertvergleich',
        'fr': 'Bland-Altman 95 % IC et comparaison des valeurs individuelles'}
    
    rnrcharts_fig_title: Dict[str, str] = {
        'en': 'Measurement system analysis',
        'de': 'Messsystemanalyse',
        'fr': 'Analyse du système de mesure'}

    rnrcharts_sub_title: Dict[str, str] = {
        'en': 'Repeatability and reproducibility (Gage R&R)',
        'de': 'Wiederholbarkeit und Reproduzierbarkeit (Gage R&R)',
        'fr': 'Répétabilité et reproductibilité (Gage R&R)'}
    
    rnrcharts_spread_proportions: Dict[str, str] = {
        'en': 'Spread proportions',
        'de': 'Streuungsanteile',
        'fr': 'Proportions de dispersion'}
    
    rnrcharts_suitability: Dict[str, str] = {
        'en': 'Suitability index Q',
        'de': 'Eignungskennwert Q',
        'fr': 'Indice d\'adéquation Q'}
    
    lm_table_caption_summary: Dict[str, str] = {
        'en': 'Model summary',
        'de': 'Modellzusammenfassung',
        'fr': 'Résumé du modèle'}
    
    lm_table_caption_statistics: Dict[str, str] = {
        'en': 'Parameter statistics',
        'de': 'Parameterstatistik',
        'fr': 'Statistiques des paramètres'}
    
    lm_table_caption_anova: Dict[str, str] = {
        'en': 'Analysis of variance',
        'de': 'Varianzanalyse',
        'fr': 'Analyse de la variance'}
    
    lm_table_caption_vif: Dict[str, str] = {
        'en': 'Variance inflation factor',
        'de': 'Varianzinflationfaktor',
        'fr': 'Facteur d\'inflation de la variance'}
    
    lm_table_caption_rnr: Dict[str, str] = {
        'en': 'Repeatability and reproducibility (R&R)',
        'de': 'Wiederholbarkeit und Reproduzierbarkeit (R&R)',
        'fr': 'Répétabilité et reproductibilité (R&R)'}
    
    lm_table_rnr_source: Dict[str, str] = {
        'en': 'Source',
        'de': 'Quelle',
        'fr': 'Source'}
    
    lm_table_caption_ref_gages: Dict[str, str] = {
        'en': 'Reference analysis',
        'de': 'Analyse der Referenzen',
        'fr': 'Analyse des références'}
    
    lm_table_caption_uncertainty: Dict[str, str] = {
        'en': 'Measurement uncertainty',
        'de': 'Messunsicherheit',
        'fr': 'Incertitude de mesure'}
    
    lm_table_caption_capabilities: Dict[str, str] = {
        'en': 'Gage capability',
        'de': 'Messsystemfähigkeit',
        'fr': 'Capacité du système de mesure'}

    _language_: Literal['en', 'de', 'fr'] = 'en'
    _username_: str = environ['USERNAME']

    @property
    def TODAY(self) -> str:
        return date.today().strftime('%Y-%m-%d')
    
    @property
    def LANGUAGE(self) -> Literal['en', 'de', 'fr']:
        """Language (abbreviation) in which the strings should be
        rendered"""
        return self._language_
    @LANGUAGE.setter
    def LANGUAGE(self, language: Literal['en', 'de', 'fr']) -> None:
        assert language in ('en', 'de', 'fr')
        self._language_ = language
    
    @property
    def USERNAME(self) -> str:
        """Username reflected in the charts in the info text, defaults 
        to username from the environment variable."""
        return self._username_
    @USERNAME.setter
    def USERNAME(self, username: str) -> None:
        self._username_ = username
    
    def __getitem__(self, item: str) -> str | Literal['']:
        _string = ''
        try:
            _string = getattr(self, item)[self.LANGUAGE]
        except AttributeError:
            warnings.warn(f'No string found for {item}!')
        except KeyError:
            warnings.warn(
                f'No string found for {item} and language {self.LANGUAGE}!')
        return _string
STR = _String_()

__all__ = ['STR']
