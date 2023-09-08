<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/entropicalabs/openqaoa/blob/main/.github/images/openqaoa_logo_offW.png" width="650">
  <img alt="OpenQAOA" src="https://github.com/entropicalabs/openqaoa/blob/main/.github/images/openqaoa_logo.png" width="650">
</picture>

  [![build test](https://github.com/entropicalabs/openqaoa/actions/workflows/test_main_linux.yml/badge.svg)](https://github.com/entropicalabs/openqaoa/actions/workflows/test_main_linux.yml)<!-- Tests (GitHub actions) -->
  [![Documentation Status](https://readthedocs.org/projects/el-openqaoa/badge/?version=latest)](https://el-openqaoa.readthedocs.io/en/latest/?badge=latest) <!-- Readthedocs -->
  [![PyPI version](https://badge.fury.io/py/openqaoa.svg)](https://badge.fury.io/py/openqaoa) <!-- PyPI -->
  [![arXiv](https://img.shields.io/badge/arXiv-2210.08695-<COLOR>.svg)](https://arxiv.org/abs/2210.08695) <!-- arXiv -->
  [![License](https://img.shields.io/pypi/l/openqaoa)](LICENSE.md)<!-- License -->
  [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)<!-- Covenant Code of conduct -->
  [![Downloads](https://pepy.tech/badge/openqaoa)](https://pepy.tech/project/openqaoa)
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/entropicalabs/openqaoa.git/main?labpath=%2Fexamples)
  [![Discord](https://img.shields.io/discord/991258119525122058)](https://discord.gg/ana76wkKBd)
  [![Website](https://img.shields.io/badge/OpenQAOA-Website-blueviolet)](https://openqaoa.entropicalabs.com/) 
</div>

# OpenQAOA

A multi-backend python library for quantum optimization using QAOA on Quantum computers and Quantum computer simulators. Check out the OpenQAOA website at [https://openqaoa.entropicalabs.com/](https://openqaoa.entropicalabs.com/)
 
**OpenQAOA is currently in OpenBeta.**

Please, consider [joining our discord](https://discord.gg/ana76wkKBd) if you want to be part of our community and participate in the OpenQAOA's development. 

## Installation instructions

OpenQAOA is divided into separately installable plugins based on the requirements of the user. The core elements of the package are placed in `openqaoa-core` which comes pre-installed with each flavour of OpenQAOA. 

Currently, OpenQAOA supports the following backends and each can be installed exclusively with the exception of `openqaoa-azure` which installs `openqaoa-qiskit` as an additional requirement because Azure backends support circuit submissions via `qiskit`.
- `openqaoa-braket` for AWS Braket
- `openqaoa-azure` for Microsoft Azure Quantum
- `openqaoa-pyquil` for Rigetti Pyquil
- `openqaoa-qiskit` for IBM Qiskit

The OpenQAOA metapackage, `openqaoa` allows you to install all OpenQAOA plug-ins together.
### Install via PyPI
You can install the latest version of OpenQAOA directly from PyPI. First, create a virtual environment with python3.8, 3.9, 3.10 and then pip install openqaoa with the following command
```
pip install openqaoa
```
### Install via git clone
Alternatively, you can install OpenQAOA manually from the GitHub repository by following the instructions below. 

**NOTE:** We recommend creating a python virtual environment for this project using a python environment manager, for instance Anaconda. Instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). Make sure to use **python 3.8** (or newer) for the environment.
1. Clone the git repository:
```
git clone https://github.com/entropicalabs/openqaoa.git
```
2. After cloning the repository `cd openqaoa` and pip install the package with instructions from the Makefile as follows
```
make local-install
```

### Installation instructions for Developers
Users can install OpenQAOA in the developer mode via the Makefile. For a clean editable install of the package run the following command from the `openqaoa` folder.
```
make dev-install
```
The package can be installed as an editable with extra requirements defined in the `setup.py`. If you would like to install the extra requirements to be able run the tests module or generate the docs, you can run the following

```
make dev-install-x,   with x = {tests, docs, all}
```
Should you face any issue during the installation, please drop us an email at openqaoa@entropicalabs.com or open an issue!

## Getting started

The API documentation for OpenQAOA can be found [here](https://el-openqaoa.readthedocs.io/en/latest/). We also provide a set of tutorials to get you started. Among the many, perhaps you can get started with the following ones:

- [Run your first OpenQAOA workflow](https://el-openqaoa.readthedocs.io/en/latest/notebooks/01_workflows_example.html)
- [How about trying some RQAOA for a change?](https://el-openqaoa.readthedocs.io/en/latest/notebooks/09_RQAOA_example.html)
- [Introducing EL's fast QAOA simulator](https://el-openqaoa.readthedocs.io/en/latest/notebooks/06_fast_qaoa_simulator.html)
- [Discover OpenQAOA's custom parametrizations](https://el-openqaoa.readthedocs.io/en/latest/notebooks/05_advanced_parameterization.html)

### Key Features

- **Build advanced QAOAs**. Create complex QAOAs by specifying custom _parametrisation_, _mixer hamiltonians_, _classical optimisers_ and execute the algorithm on either simulators or QPUs.

- **Recursive QAOA**. Run RQAOA with fully customisable schedules on simulators and QPUs alike. 

- **QPU access**. Built in access for `IBM Quantum`, `Rigetti QCS`, `Amazon Braket` and `Azure Quantum`.


### Available devives 

Devices are serviced both locally and on the cloud. For the IBM Quantum experience, the available devices depend on the specified credentials. For QCS and Amazon Braket, the available devices are listed in the table below:

| Device location | Device Name |
| --------------- | ----------- |
| local | `['qiskit.shot_simulator', 'qiskit.statevector_simulator', 'vectorized', 'pyquil.statevector_simulator']`  |
| [Amazon Braket](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html)    | IonQ, Rigetti, OQC, and simulators |
| [IBMQ](https://quantum-computing.ibm.com/)    | Please check the IBMQ backends available to your account |
| [Rigetti QCS](https://qcs.rigetti.com/sign-in)     | Aspen-11, Aspen-M-1, and QVM simulator |
| [Azure](https://azure.microsoft.com/en-us/products/quantum) | IonQ, Quantinuum, Rigetti, QCI |

## Running the tests

To run the unit-tests, first, make sure to have installed all the optional testing dependencies by running `make dev-install-tests`. Next type `pytest tests/ /src/*/tests/` from the project's root folder. This runs the common metapackage unit-tests and the unit-tests for each OpenQAOA plugin.

:warning: **Some tests require authentication**: Please, check the flags in `pytest.ini`. Currently these testes are marked `qpu`, `api`, `docker_aws`, `braket_api`, `sim`

:warning: **Some tests require authentication**: Please, note that the PyQuil-Rigetti tests contained in `test_pyquil_qvm.py` requires an active `qvm` (see Rigetti's documentation [here](https://pyquil-docs.rigetti.com/en/v3.1.0/qvm.html))
     
## Contributing and feedback

If you find any bugs or errors, have feature requests, or code you would like to contribute, feel free to open an issue or send us a pull request on GitHub.

We are always interested to hear about projects built with OpenQAOA. If you have an application you'd like to tell us about, drop us an email at [openqaoa@entropicalabs.com](mailto:openqaoa@entropicalabs.com)
