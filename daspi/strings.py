import warnings

from os import environ
from typing import Dict
from typing import Tuple
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
        'en': 'estimated kernel density',
        'de': 'geschätzte Kerndichte'}

    stripes: Dict[str, str] = {
        'en': 'lines',
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
    
    paramcharts_feature_label: Dict[str, Tuple[str, ...]] = {
        'en': ('Parameter', ''),
        'de': ('Parameter', '')}
    
    residcharts_fig_title: Dict[str, str] = {
        'en': 'Residuals Analysis',
        'de': 'Residuen Analyse'}
    
    residcharts_axes_titles: Dict[str, Tuple[str, ...]] = {
        'en': (
            'Probability for normal distribution',
            'Distribution of residuals',
            'Residuals vs. fit',
            'Residuals vs. observation'),
        'de': (
            'Wahrscheinlichkeitsnetz für Normalverteilung',
            'Verteilung der Residuen',
            'Residuen nach Anpassung',
            'Residuen nach Reihenfolge')}
    
    resid_name: Dict[str, str] = {
        'en': 'Residuals',
        'de': 'Residuen'}
    
    residcharts_feature_label: Dict[str, Tuple[str, ...]] = {
        'en': (
            'Quantiles of standard normal distribution',
            'Estimated kernel density',
            'Predicted values',
            'Observation order'),
        'de': (
            'Quantile der Standardnormalverteilung',
            'Geschätzte Kerndichte',
            'Vorhersage',
            'Beobachtungsreihenfolge')}
    
    paircharts_fig_title: Dict[str, str] = {
        'en': 'Pairwise Analysis',
        'de': 'Paarweise Analyse'}
    
    paircharts_sub_title: Dict[str, str] = {
        'en': 'Bland-Altman 95 % CI and individual value comparison',
        'de': 'Bland-Altman 95 %-KI und Einzelwertvergleich'}
    
    lm_repr_captions: Dict[str, Tuple[str, ...]] = {
        'en': (
            'Model summary',
            'Parameter statistics',
            'Analysis of variance',
            'Variance inflation factor'),
        'de': (
            'Modellzusammenfassung',
            'Parameterstatistik',
            'Varianzanalyse',
            'Varianzinflationfaktor')}

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
    
    def __getitem__(self, item:str) -> str | Tuple[str, ...] | Literal['']:
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
