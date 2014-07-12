from setuptools import setup
import os
from unshred import __version__

with open(os.path.join(os.path.dirname(__file__),
                       "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()

setup(
    name='unshred',
    version=__version__,
    url='https://github.com/dchaplinsky/unshred',
    license='MIT',
    author='Dmitry Chaplinsky',
    author_email='chaplinsky.dmitry@gmail.com',
    packages=["unshred",
              "unshred.features"],
    package_data={"unshred": ["static/*", "templates/*"]},
    description='Set of tools to analyze scanned shreds and tag them',
    platforms='any',
    install_requires=requirements,
    scripts=['unshred/split.py'],
)
