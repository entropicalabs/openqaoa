from setuptools import setup, find_packages
from os import getcwd

current_path = getcwd()

# with open("README.md", "r") as fh:
#     long_description = fh.read()

long_description = ""

with open("_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "openqaoa-core=={}".format(version),
    "openqaoa-qiskit=={}".format(version),
    "azure-quantum",
    "qdk",
    "qiskit-qir",
    "qiskit-ionq",
    "azure-quantum[qiskit]",
]

setup(
    name="openqaoa-azure",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
    packages=find_packages(where="."),
    url="https://github.com/entropicalabs/openqaoa",
    install_requires=requirements,
    license="MIT",
    description="Azure Plug-in for OpenQAOA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="quantum optimisation SDK",
)
