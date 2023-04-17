from setuptools import setup, find_namespace_packages
import os

current_path = os.getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

# Dev package will share versions with the core in it.
with open("src/openqaoa-core/openqaoa/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

core_req = [f"openqaoa-core @ file://localhost/{current_path}/src/openqaoa-core"]
qiskit_req = [f"openqaoa-qiskit @ file://localhost/{current_path}/src/openqaoa-qiskit"]
azure_req = [f"openqaoa-azure @ file://localhost/{current_path}/src/openqaoa-azure"]
pyquil_req = [f"openqaoa-pyquil @ file://localhost/{current_path}/src/openqaoa-pyquil"]
braket_req = [f"openqaoa-braket @ file://localhost/{current_path}/src/openqaoa-braket"]

setup(
    name="openqaoa",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
    packages=[],
    url="https://github.com/entropicalabs/openqaoa",
    install_requires=[],
    extras_require={
        "all": core_req + qiskit_req + azure_req + pyquil_req + braket_req
    },
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
