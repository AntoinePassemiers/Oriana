# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

import os
from setuptools import setup, find_packages


setup(
    name='oriana',
    version='1.0.0',
    description='Variational inference library',
    url='https://github.com/AntoinePassemiers/Oriana',
    author='Antoine Passemiers, Robin Petit',
    author_email='apassemi@ulb.ac.be',
    packages=['oriana'],
    include_package_data=False,
    pymodules=['oriana'],
    install_requires=[
        'numpy >= 1.13.3',
        'scipy >= 1.1.0'])
