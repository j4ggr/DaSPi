from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('daspi')
except PackageNotFoundError:
    __version__ = 'local development version'
