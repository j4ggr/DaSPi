import warnings

from os import environ
from typing import Dict
from typing import Literal
from datetime import date


class _String_:

    anderson_darling: Dict[str, str] = {
        'en': 'Anderson-Darling',
        'de': 'Anderson-Darling'}
    
    lsl: Dict[str, str] = { 
        'en': 'LSL',
        'de': 'USG'}
    
    usl: Dict[str, str] = {
        'en': 'USL',
        'de': 'OSG'}
    
    lcl: Dict[str, str] = { 
        'en': 'LCL',
        'de': 'UEG'}
    
    ucl: Dict[str, str] = {
        'en': 'UCL',
        'de': 'OEG'}
    
    excess: Dict[str, str] = {
        'en': 'excess',
        'de': 'Exzess'}
    
    skew: Dict[str, str] = {
        'en': 'skew',
        'de': 'Schiefe'}
    
    kde_ax_label: Dict[str, str] = {
        'en': 'Estimated kernel density',
        'de': 'Geschätzte Kerndichte'}

    stripes: Dict[str, str] = {
        'en': 'Lines',
        'de': 'Linien'}

    ci: Dict[str, str] = {
        'en': 'CI',
        'de': 'KI'}
    
    formula: Dict[str, str] = {
        'en': 'formula',
        'de': 'Formel'}
    
    effects_label: Dict[str, str] = {
        'en': 'Standardized effect',
        'de': 'Standardisierter Effekt'}
    
    ss_label: Dict[str, str] = {
        'en': 'Sum of Squares',
        'de': 'Summenquadrate'}
    
    paramcharts_fig_title: Dict[str, str] = {
        'en': 'Parameter Analysis',
        'de': 'Parameter Analyse'}
    
    paramcharts_sub_title: Dict[str, str] = {
        'en': 'Relative importance of parameters',
        'de': 'Relative Wichtigkeit der Parameter'}
    
    paramcharts_feature_label: Dict[str, str] = {
        'en': 'Parameter',
        'de': 'Parameter'}
    
    residcharts_fig_title: Dict[str, str] = {
        'en': 'Residuals analysis',
        'de': 'Residuen Analyse'}
    
    resid_name: Dict[str, str] = {
        'en': 'Residuals',
        'de': 'Residuen'}
    
    fit: Dict[str, str] = {
        'en': 'Fit',
        'de': 'Anpassung'}
    
    charts_flabel_quantiles: Dict[str, str]= {
        'en': 'Std. Normal Distribution quantiles',
        'de': 'Standardnormalverteilung Quantile'}
    
    charts_flabel_density: Dict[str, str] = {
        'en': 'Estimated kernel density',
        'de': 'Geschätzte Kerndichte'}
    
    charts_flabel_predicted: Dict[str, str] = {
        'en': 'Predicted values',
        'de': 'Vorhersage'}
    
    charts_flabel_observed: Dict[str, str] = {
        'en': 'Observation order',
        'de': 'Beobachtungsreihenfolge'}
    
    charts_label_alpha_th: Dict[str, str] = {
        'en': 'Threshold (α={alpha})',
        'de': 'Grenzwert (α={alpha})'}
    
    paircharts_fig_title: Dict[str, str] = {
        'en': 'Pairwise analysis',
        'de': 'Paarweise Analyse'}

    cp: Dict[str, str] = {
        'en': 'Pocess Capability index Cp',
        'de': 'Prozessfähigkeitsindex Cp'}

    cpk: Dict[str, str] = {
        'en': 'Adjusted Pocess Capability index Cpk',
        'de': 'Angepasster Prozessfähigkeitsindex Cpk'}
    
    paircharts_sub_title: Dict[str, str] = {
        'en': 'Bland-Altman 95 % CI and individual value comparison',
        'de': 'Bland-Altman 95 %-KI und Einzelwertvergleich'}
    
    lm_table_caption_summary: Dict[str, str] = {
        'en': 'Model summary',
        'de': 'Modellzusammenfassung'}
    
    lm_table_caption_statistics: Dict[str, str] = {
        'en': 'Parameter statistics',
        'de':'Parameterstatistik'}
    
    lm_table_caption_anova: Dict[str, str] = {
        'en': 'Analysis of variance',
        'de': 'Varianzanalyse'}
    
    lm_table_caption_vif: Dict[str, str] = {
        'en': 'Variance inflation factor',
        'de': 'Varianzinflationfaktor'}
    
    lm_table_caption_rnr: Dict[str, str] = {
        'en': 'Repeatability and reproducibility (R&R)',
        'de': 'Wiederholbarkeit und Reproduzierbarkeit (R&R)'}
    
    lm_table_caption_uncertainty: Dict[str, str] = {
        'en': 'Measurement uncertainty',
        'de': 'Messunsicherheit'}

    _language_: Literal['en', 'de'] = 'en'
    _username_: str = environ['USERNAME']

    @property
    def TODAY(self) -> str:
        return date.today().strftime('%Y-%m-%d')
    
    @property
    def LANGUAGE(self) -> Literal['en', 'de']:
        """Language (abbreviation) in which the strings should be
        rendered"""
        return self._language_
    @LANGUAGE.setter
    def LANGUAGE(self, language: Literal['en', 'de']) -> None:
        assert language in ('en', 'de')
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
        empty = ''
        try:
            strings = getattr(self, item)
            try:
                return strings[self.LANGUAGE]
            except KeyError:
                if isinstance(strings, dict):
                    return strings['en']
                else:
                    raise ArithmeticError
        except AttributeError:
            warnings.warn(f'No string found for {item}!')
            return empty
STR = _String_()

__all__ = ['STR']
