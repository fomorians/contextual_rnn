from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup


install_requires = [
    'tensorflow',
    'tensorflow-probability',
    'matplotlib',
]

setup(
    name='contextual_rnn',
    version='0.0.1',
    url='https://github.com/fomorians/contextual_rnn',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True)