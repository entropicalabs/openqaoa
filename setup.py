from setuptools import setup, find_packages
from os import getcwd
current_path = getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "numpy==1.22.3",
    "networkx==2.8",
    "scipy==1.8",
    "matplotlib==3.4.3",
    "qiskit==0.36.1",
    "pyquil==3.1.0",
    "pytest==7.1.0",
    "sphinx==4.5.0",
    "sphinx-autodoc-typehints==1.18.1",
    "sphinx-rtd-theme==1.0.0",
    "ipython==8.2.0" ,
    "nbconvert==6.5.0"
]

setup(
    name='openqaoa',
    version='0.0.1',
    author='Entropica Labs',
    packages=find_packages(),
    url="https://gitlab.com/entropica/openqaoa-temporary",
    description="A python SDK for Quantum Optimisation",
    license='LICENSE.md',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8+",
        "Operating System :: OS Independent"
    ],
    install_requires=requirements,
    keywords="quantum optimisation SDK"
)
