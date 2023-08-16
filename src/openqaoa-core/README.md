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

# OpenQAOA-Core

OpenQAOA is a multi-backend python library for quantum optimization using QAOA on Quantum computers and Quantum computer simulators. This package is part of a set of OpenQAOA plug-ins and forms the core functionality of the package. It includes all backend agnostic elements of the QAOA process. Along with other OpenQAOA plug-in extensions enables users to seemlessly run QAOA computations across multiple Quantum Computers. Additionally, it also comes bundled with Entropica's fast QAOA simulator called `vectorized` and a unit-depth QAOA analytical simulator (you can find more details in the Getting Started section). Check out OpenQAOA website [https://openqaoa.entropicalabs.com/](https://openqaoa.entropicalabs.com/)

**OpenQAOA is currently in OpenBeta.**

Please, consider [joining our discord](https://discord.gg/ana76wkKBd) if you want to be part of our community and participate in the OpenQAOA's development. 

## Installation instructions

### Install via PyPI

You can install the latest version of openqaoa-core directly from PyPi. We recommend creating a virtual environment with `python>=3.8` first and then simply pip install openqaoa-core with the following command.

```bash
pip install openqaoa-core
```

### Installation instructions for Developers
OpenQAOA-Core does not yet support developer install as a standalone package. If you wish to work in developer mode, please install the entire library. Instructions are available [here]()

Should you face any issue during the installation, please drop us an email at openqaoa@entropicalabs.com or open an issue!

## Getting started

The documentation for OpenQAOA-Core can be found [here](https://el-openqaoa.readthedocs.io/en/latest/).

We also provide a set of tutorials to get you started. Among the many, perhaps you can get started with the following ones:
- A notebook showing Analytical Simulator usage
- [Introducing EL's fast QAOA simulator](https://el-openqaoa.readthedocs.io/en/latest/notebooks/06_fast_qaoa_simulator.html)
- [Discover OpenQAOA's custom parametrizations](https://el-openqaoa.readthedocs.io/en/latest/notebooks/05_advanced_parameterization.html)
- Notebook demonstrating different optimizers to choose from

### Key Features

- **Build advanced QAOAs**. Create complex QAOAs by specifying custom _parametrisation_, _mixer hamiltonians_, _classical optimisers_ and execute the algorithm on either simulators or QPUs.

- **Recursive QAOA**. Run RQAOA with fully customisable schedules on simulators and QPUs alike. 

- **QPU access**. Built in access for `IBM Quantum`, `Rigetti QCS`, `Amazon Braket` and `Azure Quantum` (Please install the OpenQAOA plugin corresponding to the QPU platform for running QAOA on these devices). 

### Available devives 

Devices are serviced both locally and on the cloud. For the IBM Quantum experience, the available devices depend on the specified credentials. For QCS and Amazon Braket, the available devices are listed in the table below:

| Device location | Device Name |
| --------------- | ----------- |
| `local` | `['vectorized', 'analytical_simulator']` |

## Running the tests

TODO

## Contributing and feedback

If you find any bugs or errors, have feature requests, or code you would like to contribute, feel free to open an issue or send us a pull request on GitHub.

We are always interested to hear about projects built with EntropicaQAOA. If you have an application you'd like to tell us about, drop us an email at openqaoa@entropicalabs.com.
