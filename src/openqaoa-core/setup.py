from setuptools import setup, find_packages
from os import getcwd

current_path = getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("openqaoa/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

requirements_docs = open("requirements_docs.txt").readlines()
requirements_docs = [r.strip() for r in requirements_docs]

requirements_test = open("requirements_test.txt").readlines()
requirements_test = [r.strip() for r in requirements_test]

setup(
    name="openqaoa-core",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
    packages=find_packages(where="."),
    entry_points={"openqaoa.plugins": []},
    url="https://github.com/entropicalabs/openqaoa",
    install_requires=requirements,
    license="MIT",
    description="OpenQAOA is a python open-source multi-backend Software Development Kit to create, customise and execute the Quantum Approximate Optimisation Algorithm (QAOA) on Noisy Intermediate-Scale Quantum (NISQ) devices, and simulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="quantum optimisation SDK",
    extras_require={
        "docs": requirements_docs,
        "tests": requirements_test,
        "all": requirements_docs + requirements_test,
    },
)
