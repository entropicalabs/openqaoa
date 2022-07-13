 <div align="center">

  <!-- OpenQAOA logo -->
  <a href="https://github.com/entropicalabs/openqaoa"><img src=".github/images/openqaoa_logo.png?raw=true" alt="OpenQAOA logo" width="300"/></a>

#

  [![build test](https://github.com/entropicalabs/openqaoa/actions/workflows/test.yml/badge.svg)](https://github.com/entropicalabs/openqaoa/actions/workflows/test.yml)<!-- Tests (GitHub actions) -->
  [![Documentation Status](https://readthedocs.com/projects/entropica-labs-openqaoa/badge/?version=latest&token=bdaaa98247ceb7e8e4bbd257d664fa3cc42ab6f4588ddabbe5afa6a3d20a1008)](https://entropica-labs-openqaoa.readthedocs-hosted.com/en/latest/?badge=latest) <!-- Readthedocs -->
   [![License](https://img.shields.io/badge/%F0%9F%AA%AA%20license-Apache%20License%202.0-lightgrey)](LICENSE.md)<!-- License -->
 [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)<!-- Covenant Code of conduct -->
 [![codecov](https://codecov.io/gh/entropicalabs/openqaoa/branch/dev/graph/badge.svg?token=ZXD77KM5OR)](https://codecov.io/gh/entropicalabs/openqaoa) <!-- Code coverage -->
</div>

# OpenQAOA

A multi-backend python library for quantum optimization using QAOA on Quantum computers and Quantum computer simulators.

OpenQAOA is currently in OpenBeta.

## Installation instructions

1. Clone the git repository:

```bash
git clone git@github.com:entropicalabs/openqaoa.git
```

2. Creating a python `virtual environment` for this project is recommended. (for instance, using conda). Instructions on how to create a virtual environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). Make sure to use **python 3.8.8** for the environment.

3. After cloning the repository `cd openqaoa` and pip install in edit mode. Use the following command for a vanilla install with the `scipy` optimizers:

```bash
pip install -e .
```

## Getting started

The documentation for OpenQAOA ca be found [here](https://entropica-labs-openqaoa.readthedocs-hosted.com/en/latest/)


### Key Features

- *Build advanced QAOAs*. Create complex QAOAs by specifying custom _parametrisation_, _mixer hamiltonians_, _classical optimisers_ and execute the algorithm on either simulators or QPUs.

- *Recursive QAOA*. Run RQAOA with fully customisable schedules on simulators and QPUs alike. 

- *QPU access*. Built in access for `IBMQ`, `Rigeti QCS`, and `AWS`.


### Available backend 

Currently, the available devices are:

| Device location  | Device Name |
| ------------- | ------------- |
| `local`  | `['qiskit.shot_simulator', 'qiskit.statevector_simulator', 'qiskit.qasm_simulator', 'vectorized', 'pyquil.statevector_simulator']`  |
| `ibmq`    | Please check the IMBQ backends available to your account |
| `qcs`     | `[nq-qvm, Aspen-11, Aspen-M-1]`


## Running the tests

To run the test just type `pytest tests/.` from the project's root folder. Bear in mind that `test_pyquil_qvm.py` requires an active `qvm` (see righetti's documentation [here](https://pyquil-docs.rigetti.com/en/v3.1.0/qvm.html)), and `test_qpu_qiskit.py` and `test_qpu_auth.py` require a valid IBMQ token in the file `tests/credentials.json`

## Contributing and feedback

If you find any bugs or errors, have feature requests, or code you would like to contribute, feel free to open an issue or send us a pull request on GitHub.

We are always interested to hear about projects built with EntropicaQAOA. If you have an application you’d like to tell us about, drop us an email at openqaoa@entropicalabs.com.