from setuptools import setup, find_packages
from os import getcwd

current_path = getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("openqaoa/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "amazon-braket-sdk",
    "pandas==1.1.4",
    "sympy>=1.10.1",
    "numpy==1.19.5",
    "networkx==2.4",
    "matplotlib==3.2.2",
    "scipy==1.5.2",
    "qiskit-terra",
    "qiskit-aer",
    "qiskit-ibmq-provider",
    "docplex>=2.23.1",
    "autograd>=1.4",
    "semantic_version>=2.10",
    "autoray>=0.3.1"
]

requirements_docs = [
    "sphinx==4.5.0",
    "sphinx-autodoc-typehints==1.18.1",
    "sphinx-rtd-theme==1.0.0",
    "nbsphinx==0.8.9",
    "ipython==8.2.0",
    "nbconvert==6.0.7"    
]

requirements_test = [
    "pytest>=7.1.0",
    "pytest-cov>=3.0.0",
    "ipython>=8.2.0",
    "nbconvert>=6.5.1",
    "plotly>=5.9.0",
    "cplex>=22.1.0.0",
    "ipykernel>6.0.0"
]

setup(
    name="openqaoa",
    python_requires='>=3.7, <3.11',
    version= version,
    author="Entropica Labs",
    packages=find_packages(where="."),
    url="https://github.com/entropicalabs/openqaoa",
    install_requires= requirements,
    license="Apache 2.0",
    description= "OpenQAOA is a python open-source multi-backend Software Development Kit to create, customise and execute the Quantum Approximate Optimisation Algorithm (QAOA) on Noisy Intermediate-Scale Quantum (NISQ) devices, and simulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="quantum optimisation SDK",
    extras_require = {
        "docs":requirements_docs,
        "tests":requirements_test,
        "all": requirements_docs + requirements_test
    },

)
