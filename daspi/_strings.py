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


    LANGUAGE: str = environ.get('DASPI_LANGUAGE', 'en')
    USERNAME: str = environ['USERNAME']

    @property
    def TODAY(self):
        return date.today().strftime('%Y.%m.%d')
    
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
