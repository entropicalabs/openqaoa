from setuptools import setup, find_packages
from os import getcwd

current_path = getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "sympy>=1.10.1",
    "numpy>=1.22.3",
    "networkx>=2.8",
    "scipy==1.8",
    "matplotlib>=3.4.3",
    "qiskit>=0.36.1",
    "pyquil>=3.1.0"
]

requirements_docs = [
    "sphinx==4.5.0",
    "sphinx-autodoc-typehints==1.18.1",
    "sphinx-rtd-theme==1.0.0",
    "nbsphinx==0.8.9",
    "ipython==8.2.0",
    "nbconvert>=6.5.1"    
]

requirements_test = [
    "pytest==7.1.0",
    "pytest-cov==3.0.0",
    "ipython==8.2.0",
    "nbconvert>=6.5.1"
]

setup(
    name="openqaoa",
    version="0.0.1",
    author="Entropica Labs",
    packages=find_packages(),
    url="https://github.com/entropicalabs/openqaoa",
    description="A python SDK for Quantum Optimisation",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    keywords="quantum optimisation SDK",
    extras_require = {
        "docs":requirements_docs,
        "tests":requirements_test,
        "all": requirements_docs + requirements_test
    },

)
