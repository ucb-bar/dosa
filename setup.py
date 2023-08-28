#!/usr/bin/env python
from setuptools import setup, find_packages

requirements = [
    "scikit-learn==1.1.1",
    "pandas==1.5.1",
    "numpy==1.22.3",
    "torch==1.12.1",
    "swifter==1.3.3",
    "gurobipy==9.5.2",
    "matplotlib==3.5.2",
]

setup(
    # Metadata
    # install_requires=requirements,
    packages=find_packages()
)
