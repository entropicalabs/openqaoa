<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/entropicalabs/openqaoa/blob/main/.github/images/openqaoa_logo_offW.png" width="650">
  <img alt="OpenQAOA" src="https://github.com/entropicalabs/openqaoa/blob/main/.github/images/openqaoa_logo.png" width="650">
</picture>

[![build test](https://github.com/entropicalabs/openqaoa/actions/workflows/test_main_linux.yml/badge.svg)](https://github.com/entropicalabs/openqaoa/actions/workflows/test_main_linux.yml)<!-- Tests (GitHub actions) -->
[![Documentation Status](https://readthedocs.org/projects/el-openqaoa/badge/?version=latest)](https://el-openqaoa.readthedocs.io/en/latest/?badge=latest)

<!-- Readthedocs -->

[![PyPI version](https://badge.fury.io/py/openqaoa.svg)](https://badge.fury.io/py/openqaoa)

<!-- PyPI -->

[![arXiv](https://img.shields.io/badge/arXiv-2210.08695-<COLOR>.svg)](https://arxiv.org/abs/2210.08695)

<!-- arXiv -->

[![License](https://img.shields.io/pypi/l/openqaoa)](LICENSE.md)<!-- License -->
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)<!-- Covenant Code of conduct -->
[![Downloads](https://pepy.tech/badge/openqaoa)](https://pepy.tech/project/openqaoa)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/entropicalabs/openqaoa.git/main?labpath=%2Fexamples)
[![Discord](https://img.shields.io/discord/991258119525122058)](https://discord.gg/ana76wkKBd)
[![Website](https://img.shields.io/badge/OpenQAOA-Website-blueviolet)](https://openqaoa.entropicalabs.com/)

</div>

# OpenQAOA-Pyquil Plugin

OpenQAOA is a multi-backend python library for quantum optimization using QAOA
on Quantum computers and Quantum computer simulators. This package is part of a
set of OpenQAOA plug-ins that lets users run QAOA computations on Rigetti's
devices accessible through QCS, and devices that support quil instructions.
Check out OpenQAOA website
[https://openqaoa.entropicalabs.com/](https://openqaoa.entropicalabs.com/)

**OpenQAOA is currently in OpenBeta.**

Please, consider [joining our discord](https://discord.gg/ana76wkKBd) if you
want to be part of our community and participate in the OpenQAOA's development.

## Installation instructions

### Install via PyPI

You can install the latest version of openqaoa-pyquil directly from PyPi. We
recommend creating a virtual environment with `python>=3.9` first and then
simply pip install openqaoa-pyquil with the following command.

**NOTE:** Installing `openqaoa-pyquil` installs `openqaoa-core` by default

```bash
pip install openqaoa-pyquil
```

### Installation instructions for Developers

OpenQAOA-Qiskit does not yet support developer install as a standalone package.
If you wish to work in developer mode, please install the entire library.
Instructions are available [here]()

Should you face any issue during the installation, please drop us an email at
openqaoa@entropicalabs.com or open an issue!

## Getting started

The documentation for OpenQAOA-Pyquil can be found
[here](https://el-openqaoa.readthedocs.io/en/latest/).

We also provide a set of tutorials to get you started. Among the many, perhaps
you can get started with the following ones:

- Link OpenQAOA Pyquil notebooks
- Using QPU
- Using as QVM

### Available devives

OpenQAOA-Pyquil services devices both locally and on the cloud. The Rigetti
supports access to their QPUs through the QCS platform. Moreover, users can also
access a `qvm` emulator of Rigetti QPUs that runs locally. **NOTE:** Using a QVM
requires installation of Rigetti Forest SDK that does not come pre-installed
with OpenQAOA. If required the user may install the SDK
[here](https://qcs.rigetti.com/sdk-downloads)

| Device location                                | Device Name                            |
| ---------------------------------------------- | -------------------------------------- |
| [Rigetti QCS](https://qcs.rigetti.com/sign-in) | Aspen-11, Aspen-M-1, and QVM simulator |

## Running the tests

TODO

## Contributing and feedback

If you find any bugs or errors, have feature requests, or code you would like to
contribute, feel free to open an issue or send us a pull request on GitHub.

We are always interested to hear about projects built with EntropicaQAOA. If you
have an application you'd like to tell us about, drop us an email at
openqaoa@entropicalabs.com.
