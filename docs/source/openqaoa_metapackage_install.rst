OpenQAOA Metapackage Installation
=================================

The following instructions install OpenQAOA along with all optional plugins

OpenQAOA is divided into separately installable plugins based on the requirements of the user. The core elements of the package are placed in `openqaoa-core` which comes pre-installed with each flavour of OpenQAOA. 

Currently, OpenQAOA supports the following backends and each can be installed exclusively with the exception of `openqaoa-azure` which installs `openqaoa-qiskit` as an additional requirement because Azure backends support circuit submissions via `qiskit`.
- `openqaoa-braket` for AWS Braket
- `openqaoa-azure` for Microsoft Azure Quantum
- `openqaoa-pyquil` for Rigetti Pyquil
- `openqaoa-qiskit` for IBM Qiskit

The OpenQAOA metapackage allows you to install all OpenQAOA plug-ins together.

Install via PyPI
----------------
You can install the latest version of OpenQAOA directly from PyPI. First, create a virtual environment with python3.8, 3.9, 3.10 and then pip install openqaoa with the following command
```
pip install openqaoa
```

Install via git clone
---------------------
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

Installation instructions for Developers
----------------------------------------
Users can install OpenQAOA in the developer mode via the Makefile. For a clean editable install of the package run the following command from the `openqaoa` folder.
```
make dev-install
```
The package can be installed as an editable with extra requirements defined in the `setup.py`. If you would like to install the extra requirements to be able run the tests module or generate the docs, you can run the following

```
make dev-install-x,   with x = {tests, docs, all}
```

Should you face any issue during the installation, please drop us an email at openqaoa@entropicalabs.com or open an issue!