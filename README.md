 <div align="center">

  <!-- OpenQAOA logo -->
  <a href="https://github.com/entropicalabs/openqaoa"><img src=".github/images/openqaoa_logo.png?raw=true" alt="OpenQAOA logo" width="300"/></a>

#
  [![build test](https://github.com/entropicalabs/openqaoa/actions/workflows/test_main_linux.yml/badge.svg)](https://github.com/entropicalabs/openqaoa/actions/workflows/test_main_linux.yml)<!-- Tests (GitHub actions) -->
  [![Documentation Status](https://readthedocs.org/projects/el-openqaoa/badge/?version=latest)](https://el-openqaoa.readthedocs.io/en/latest/?badge=latest) <!-- Readthedocs -->
  [![License](https://img.shields.io/badge/%F0%9F%AA%AA%20license-Apache%20License%202.0-lightgrey)](LICENSE.md)<!-- License -->
  [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)<!-- Covenant Code of conduct -->
  [![codecov](https://codecov.io/gh/entropicalabs/openqaoa/branch/dev/graph/badge.svg?token=ZXD77KM5OR)](https://codecov.io/gh/entropicalabs/openqaoa) <!-- Code coverage -->
</div>

# OpenQAOA

A multi-backend python library for quantum optimization using QAOA on Quantum computers and Quantum computer simulators.

OpenQAOA is currently in OpenBeta. 

Please, consider [joining our discord](https://discord.gg/ana76wkKBd) if you want to be part of our community and participate in the OpenQAOA's development. 

## Installation instructions

You can install the latest version of OpenQAOA directly from PyPi. First, create a virtual environment with python3.8+ and then simply pip install openqaoa with the following command

```bash
pip install openqaoa
```

Alternatively, you can install manually directly from the GitHub repository by

1. Clone the git repository:

```bash
git clone git@github.com:entropicalabs/openqaoa.git
```

2. Creating a python `virtual environment` for this project is recommended. (for instance, using conda). Instructions on how to create a virtual environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). Make sure to use **python 3.8** for the environment.

3. After cloning the repository `cd openqaoa` and pip install in edit mode. Use the following command for a vanilla install with the `scipy` optimizers:

```bash
pip install -e .
```

Should you face any issue during the installation, please drop us an email at openqaoa@entropicalabs.com or open an issue!

## Getting started

The documentation for OpenQAOA can be found [here](https://el-openqaoa.readthedocs.io/en/latest/).

We also provide a set of tutorials to get you started. Among the many, perhaps you can get started with the following ones:

- [Run your first OpenQAOA workflow](https://el-openqaoa.readthedocs.io/en/latest/notebooks/1_workflows_example.html)
- [How about trying some RQAOA for a change?](https://el-openqaoa.readthedocs.io/en/latest/notebooks/9_RQAOA_example.html)
- [Introducing EL's fast QAOA simulator](https://el-openqaoa.readthedocs.io/en/latest/notebooks/6_fast_qaoa_simulator.html)
- [Discover OpenQAOA's custom parametrizations](https://el-openqaoa.readthedocs.io/en/latest/notebooks/5_advanced_parameterization.html)

### Key Features

- **Build advanced QAOAs**. Create complex QAOAs by specifying custom _parametrisation_, _mixer hamiltonians_, _classical optimisers_ and execute the algorithm on either simulators or QPUs.

- **Recursive QAOA**. Run RQAOA with fully customisable schedules on simulators and QPUs alike. 

- **QPU access**. Built in access for `IBMQ`, `Rigetti QCS`, and `AWS`.


### Available backend 

Currently, the available devices are:

| Device location  | Device Name |
| ------------- | ------------- |
| `local`  | `['qiskit.shot_simulator', 'qiskit.statevector_simulator', 'qiskit.qasm_simulator', 'vectorized', 'pyquil.statevector_simulator']`  |
| `ibmq`    | Please check the IBMQ backends available to your account |
| `qcs`     | `[nq-qvm, Aspen-11, Aspen-M-1]`


## Running the tests

To run the test, first, make sure to have installed all the optional testing dependencies by running `pip install .[tests]` (note, the braket must to be escaped if you are using the popular zsh shell), and then just type `pytest tests/.` from the project's root folder. Bear in mind that `test_pyquil_qvm.py` requires an active `qvm` (see Rigetti's documentation [here](https://pyquil-docs.rigetti.com/en/v3.1.0/qvm.html)), and `test_qpu_qiskit.py` and `test_qpu_auth.py` require a valid IBMQ token in the file `tests/credentials.json`.

## Contributing and feedback

If you find any bugs or errors, have feature requests, or code you would like to contribute, feel free to open an issue or send us a pull request on GitHub.

We are always interested to hear about projects built with EntropicaQAOA. If you have an application you'd like to tell us about, drop us an email at openqaoa@entropicalabs.com.
