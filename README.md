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

A multi-backend python library for quantum optimization usig QAOA on Quantum computers and Quantum computer simulators.

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

## Building the docs

If you have installed OpenQAOA using the setup file then all the required libraries to build the docs should already be in place. However, if something went wrong they can be easily installed by running the following command

```bash
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme
```

Then, simply navigate to the `docs` folder by typing `cd docs/` and simply type

```bash
make html
```

and the docs should appear in the folder `docs/build/html`, and can be opened by simply opening the `docs/build/html/index.html` file.

## Getting started

There are two ways to solve optimizations problems using OpenQAOA.

### Workflows

#### QAOA

Workflows are a simplified way to run end to end QAOA or RQAOA. In their basic format they consist of the following steps.

A reference jupyter notebook can be found [here](examples/Workflows_example.ipynb)

First, create a problem instance. For example, an instance of vertex cover:

```python
from openqaoa.problems.problem import MinimumVertexCover
import networkx
g = networkx.circulant_graph(6, [1])
vc = MinimumVertexCover(g, field =1.0, penalty=10)
qubo_problem = vc.get_qubo_problem()
```

Where [networkx](https://networkx.org/) is an open source Python package that can easily, among other things, create graphs.

```python
from openqaoa.workflows.optimizer import QAOA  
q = QAOA()
q.compile(qubo_problem)
q.optimize()
```

Once the binary problem is defined, the simplest workflow can be defined as

```python
from openqaoa.workflows.optimizer import QAOA  
q = QAOA()
q.compile(qubo_problem)
q.optimize() 
```

Workflows can be customised using some convenient setter functions. First, we need to set the device where we want to execute the workflow

```python
from openqaoa.devices import create_device
qcs_credentials = {'as_qvm':True, 'execution_timeout' : 10, 'compiler_timeout':10}
device = create_device(location='qcs',name='6q-qvm',**qcs_credentials)
```

Then, the QAOA parameters can be set as follow

```python
q_custom = QAOA()
q_custom.set_circuit_properties(p=10, param_type='extended', init_type='ramp', mixer_hamiltonian='x')
q_custom.set_device(device)
q_custom.set_backend_properties(n_shot=200, cvar_alpha=1)
q_custom.set_classical_optimizer(method='nelder-mead', maxiter=2)
q_custom.compile(qubo_problem)
q_custom.optimize()
```

Currently, the available devices are:

| Device location  | Device Name |
| ------------- | ------------- |
| `local`  | `['qiskit.shot_simulator', 'qiskit.statevec_simulator', 'qiskit.qasm_simulator', 'vectorized', 'pyquil.statevector_simulator']`  |
| `ibmq`    | Please check the IMBQ backends available to your account |
| `qcs`     | `[nq-qvm, Aspen-11, Aspen-M-1]`

With the notation `nq-qvm` it is intended that `n` is a positive integer. For example, `6q-qvm`.

The `vectorised` backend is developed by Entropica Labs and works by targeting active qubits (on which gates are to be applied in any given Hamiltonian term) by using the numpy slicing operators, and applying the gate operations in place. This allows the operators and their action on the wavefunction to be constructed and performed in a simple and fast way.

Note that in order to use the Rigetti devices you either need to be running your code on Rigetti's [Quantum Cloud Services](https://qcs.rigetti.com/sign-in) or, in case you want to run it locally from your machine, start qvm and quilc. More information on how to start them can be found in https://docs.rigetti.com/qcs/getting-started.

#### Recursive QAOA

A more cohmprensive notebook is [RQAOA_example](examples/RQAOA_example.ipynb)

```python
from openqaoa.workflows.optimizer import RQAOA
r = RQAOA(rqaoa_type='adaptive')
r.set_rqaoa_parameters(n_max=5, n_cutoff = 5)
r.compile(qubo_problem)
r.optimize()
```

rqaoa_type can take two values which select elimination strategies. The user can choose between `adaptive` or `custom`.

### Factory mode

The user is also free to directly access the source code without using the workflow API.

* [comparing vectorized, pyquil, and qiskit backents](examples/test_backends_correctness.ipynb)
* [Parameter sweep for vectorised](examples/openqaoa_example_vectorised.ipynb)

The basic procedure is the following

First, import all the necessay functions

```python
from openqaoa.qaoa_parameters import Hamiltonian, QAOACircuitParams, create_qaoa_variational_params
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.devices import DevicePyquil, create_device
from openqaoa.optimizers.qaoa_optimizer import ScipyOptimizer
```

Then specify terms and weights in order to define the cost hamiltonian

```python
terms = [(1,2),(2,3),(0,3),(4,0),(1,),(3,)]
coeffs = [1,2,3,4,3,5]
n_qubits = 5
cost_hamil = Hamiltonian.classical_hamiltonian(terms=terms,coeffs=coeffs,constant=0)
mixer_hamil = X_mixer_hamiltonian(n_qubits=n_qubits)
```

After having created the hamiltonians it is time to create the Circuit parameters and the Variational Parameters

```python
qaoa_circuit_params = QAOACircuitParams(cost_hamil,mixer_hamil,p=1)
params = create_qaoa_variational_params(qaoa_circuit_params, params_type='fourier',init_type='rand',q=1)
```

Then proceed by instantiating the backend device

```python
device_pyquil = create_device('qcs',"Aspen-11", as_qvm=True, execution_timeout = 10, compiler_timeout=10)
backend_obj_pyquil = get_qaoa_backend(circuit_params, device_pyquil, n_shots=1000)
```

And finally, create the classical optimizer and minimize the objective function

```python
optimizer_dict = {'method': 'cobyla', 'maxiter': 10}
optimizer_obj = ScipyOptimizer(backend_obj, params, optimizer_dict)
optimizer_obj()
```

The result of the optimization will the be accessible as

```python
optimizer_obj.results_information()
```

## Running the tests

To run the test just type `pytest tests/.` from the project's root folder. Bear in mind that `test_pyquil_qvm.py` requires an active `qvm`, and `test_qpu_qiskit.py` and `test_qpu_auth.py` require a valid IBMQ token in the file `tests/credentials.json`
