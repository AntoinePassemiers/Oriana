# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup


packages = [
        'oriana',
        'oriana.singlecell',
        'oriana.models',
        'oriana.nodes',
        'oriana.nodes.deterministic',
        'oriana.nodes.probabilistic']

setup(
    name='oriana',
    version='1.0.0',
    description='Variational inference library',
    url='https://github.com/AntoinePassemiers/Oriana',
    author='Antoine Passemiers, Robin Petit',
    author_email='apassemi@ulb.ac.be',
    packages=packages,
    include_package_data=False,
    install_requires=[
        'numpy >= 1.13.3',
        'scipy >= 1.1.0'])
