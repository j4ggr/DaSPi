from importlib.metadata import version
from importlib.metadata import PackageNotFoundError

try:
    __version__ = version('daspi')
except PackageNotFoundError:
    __version__ = 'local development version'
