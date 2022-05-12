# OpenQAOA

Multi-backend SDK for quantum optimization


## Installation instructions

1. Clone the git repository:

```bash
git clone git@github.com:entropicalabs/openqaoa.git
```

2. Creating a python `virtual enviorement` for this project is recommended. (for instance, using conda). Instructions on how to create a virtual environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). Make sure to use **python 3.8.8** for the environment.

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
```
from openqaoa.problems.problem import MinimumVertexCover
g = networkx.circulant_graph(6, [1])
vc = MinimumVertexCover(g, field =1.0, penalty=10)
pubo_problem = vc.get_pubo_problem()
```

Where [networkx](https://networkx.org/) is an open source Python package that can easily, among other things, create graphs.

Once the binary problem is defined, the simplest workflow can be defined as 

```  
from openqaoa.workflows.optimizer import QAOA  
q = QAOA()
q.compile(pubo_problem)
q.optimize() 
```

Workflows can be customised using some convenient setter functions

```
q_custom = QAOA()
q_custom.set_circuit_properties(p=10, param_type='extended', init_type='ramp', mixer_hamiltonian='x')
q_custom.set_device_properties(device_location='qcs', device_name='Aspen-11', cloud_credentials={'name' : "Aspen11", 'as_qvm':True, 'execution_timeout' : 10, 'compiler_timeout':10})
q_custom.set_backend_properties(n_shots=200, cvar_alpha=1)
q_custom.set_classical_optimizer(method='nelder-mead', maxiter=2)
q_custom.compile(pubo_problem)
q_custom.optimize()
```

Currently, the available devices are:

| Device location  | Device Name |
| ------------- | ------------- |
| `local`  | `['qiskit_shot_simulator', 'qiskit_statevec_simulator', 'qiskit_qasm_simulator', 'vectorized', 'nq-qvm', 'Aspen-11']`  |
| `IBMQ`    | Please check the IMBQ backends available to your account |
| `QCS`     | `[nq-qvm, Aspen-11, Aspen-M-1]`

With the notation `nq-qvm` it is intended that `n` is a positive integer. For example, `6q-qvm`.

The `vectorised` backend is developed by Entropica Labs and works by permuting active qubits (on which gates are to be applied in any given Hamiltonian term) to the beginning of the register, applying the gate, then undoing the permutation. This allows the operators and their action on the wavefunction to be constructed and performed in a simple, canonical way.

#### Recursive QAOA

A more cohmprensive notebook is [RQAOA_example](examples/RQAOA_example.ipynb)

```
from openqaoa.workflows.optimizer import RQAOA
r = RQAOA(rqaoa_type='adaptive')
r.set_rqaoa_parameters(n_max=5, n_cutoff = 5)
r.compile(pubo_problem)
r.optimize()
```

rqaoa_type can take two values which select elimination strategies. The user can choose between `adaptive` or `custom`.


### Factory mode

The user is also free to directly access the source code without using the workflow API. 

A few reference notebooks can be found:
* [comparing vectorized, pyquil, and qiskit backents](examples/test_backends_correctness.ipynb)
* [Parameter sweep for vectorised](examples/openqaoa_example_vectorised.ipynb)


The basic procedure is the following

First, specify terms and weights in order to define the cost hamiltonian

```
terms = [(1,2),(2,3),(0,3),(4,0),(1,),(3,)]
coeffs = [1,2,3,4,3,5]

cost_hamil = Hamiltonian.classical_hamiltonian(terms=terms,coeffs=coeffs,constant=0)
mixer_hamil = X_mixer_hamiltonian(n_qubits=5)
```

After having created the hamiltonians it is time to create the Circuit parameters and the Variational Parameters
```
qaoa_circuit_params = QAOACircuitParams(cost_hamil,mixer_hamil,p=1)
params = create_qaoa_variational_params(qaoa_circuit_params, param_type='fourier',init_type='rand',q=1)
```

Then proceed by instantiating the backend device

```
backend_obj = QAOAvectorizedBackendSimulator(circuit_params = circuit_params, append_state = None, prepend_state = None, init_hadamard = True)

```

And finally, create the classical optimizer and minimize the objective function

```
optimizer_dict = {'method': 'cobyla', 'maxiter': 10}
optimizer_obj = ScipyOptimizer(backend_obj, variate_params, optimizer_dict)
optimizer_obj()
```

The result of the optimization will the be accessible as 
```
optimizer_obj.results_information()
```


## Running the tests

To run the test just type `pytest tests/.` from the project's root folder. Bear in mind that `test_pyquil_qvm.py` requires an active `qvm`, and `test_qpu_qiskit.py` and `test_qpu_auth.py` require a valid IBMQ token in the file `tests/credentials.json`
