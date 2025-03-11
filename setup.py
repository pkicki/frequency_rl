from setuptools import setup, find_packages
from frequency_rl import __version__



setup(author="Piotr Kicki",
      url="https://github.com/pkicki/frequency_rl",
      version=__version__,
      packages=[package for package in find_packages()
                if package.startswith('frequency_rl')],
      )
