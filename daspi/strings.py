import warnings

from os import environ
from typing import Literal
from datetime import date


class _String_:

    anderson_darling: dict = {
        'en': 'Anderson-Darling',
        'de': 'Anderson-Darling'}
    
    lsl: dict = { 
        'en': 'LSL',
        'de': 'USG'}
    
    usl: dict = {
        'en': 'USL',
        'de': 'OSG'}
    
    lcl: dict = { 
        'en': 'LCL',
        'de': 'UEG'}
    
    ucl: dict = {
        'en': 'UCL',
        'de': 'OEG'}
    
    excess: dict = {
        'en': 'excess',
        'de': 'Exzess'}
    
    skew: dict = {
        'en': 'skew',
        'de': 'Schiefe'}
    
    kde_ax_label: dict = {
        'en': 'estimated kernel density',
        'de': 'geschätzte Kerndichte'}

    stripes: dict = {
        'en': 'lines',
        'de': 'Linien'}

    ci: dict = {
        'en': 'CI',
        'de': 'KI'}

    _language_: str = 'en'
    _username_: str = environ['USERNAME']

    @property
    def TODAY(self):
        return date.today().strftime('%Y.%m.%d')
    
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
    
    def __getitem__(self, item:str) -> str | Literal['']:
        empty = ''
        try:
            strings = getattr(self, item)
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
