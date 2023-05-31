from setuptools import setup, find_packages
from os import getcwd

current_path = getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("openqaoa_qiskit/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "openqaoa-core=={}".format(version),
]

add_requirements = open('requirements.txt').readlines()
add_requirements = [r.strip() for r in add_requirements]

requirements.extend(add_requirements)

setup(
    name="openqaoa-qiskit",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
    packages=find_packages(where="."),
    entry_points={
        "openqaoa.plugins": [
            "qiskit = openqaoa_qiskit.backend_config"
        ]
    },
    url="https://github.com/entropicalabs/openqaoa",
    install_requires=requirements,
    license="MIT",
    description="Qiskit Plug-in for OpenQAOA",
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
