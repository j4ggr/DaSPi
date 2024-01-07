from os import environ
from typing import Literal
from datetime import date

LANG = environ.get('DASPI_LANGUAGE', 'en')

class _String_:
    USERNAME: str = environ['USERNAME']
    @property
    def TODAY(self):
        return date.today().strftime('%Y.%m.%d')
    
    anderson_darling: dict = {
        'en': 'Anderson-Darling'}
    
    def __getitem__(self, item:str) -> str | Literal['']:
        try:
            txt = getattr(self, item)
            return txt[LANG]
        except KeyError:
            return txt['en']
        except AttributeError:
            return ''
STR = _String_()

__all__ = [
    'LANG',
    STR.__name__]
