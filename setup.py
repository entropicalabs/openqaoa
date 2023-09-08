from setuptools import setup, find_namespace_packages
import os

current_path = os.getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

# Dev package will share versions with the core in it.
with open("_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    f"{each_folder_name}=={version}"
    for each_folder_name in os.listdir("src")
    if "openqaoa-" in each_folder_name
]

setup(
    name="openqaoa",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
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
)
