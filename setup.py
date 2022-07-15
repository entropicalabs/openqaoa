from setuptools import setup, find_packages
from os import getcwd
current_path = getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "amazon-braket-sdk==1.23.0",
    "sympy>=1.9",
    "numpy>=1.19",
    "networkx>=2.6",
    "scipy>=1.7.1",
    "matplotlib>=3.0.3",
    "qiskit>=0.36.1",
    "pyquil>=3.1.0"
]

setup(
    name='openqaoa',
    version='0.0.1',
    author='Entropica Labs',
    packages=find_packages(),
    url="https://github.com/entropicalabs/openqaoa",
    description="A python SDK for Quantum Optimisation",
    license='LICENSE.md',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.7+",
        "Operating System :: OS Independent"
    ],
    install_requires=requirements,
    keywords="quantum optimisation SDK"
)
