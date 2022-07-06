Welcome to OpenQAOA's documentation!
====================================

OpenQAOA is an advanced multi-backend SDK for quantum optimization designed to ease research efforts within the VQA enviorement while ensuring reliability and reproducibility of results


Features
--------

Key features of OpenQAOA:

* Simple yet customisable workflows for QAOA and RQAOA deployable on
   * IMBQ devices
   * Rigettis' Quantum Clound Services
   * AWS's Braket
   * Local simulators (including Rigettis' QVM, IBM's Qiskit, and Entropica Labs' vectorized simulator)
* Multiple parametriation strategies:
   * Standard, Fourier, and Annealing
   * Each class can further controlled by selecting standard or extended parameter configurations
* Multiple Initliaisation strategies:
   * Linear ramp, random, and custom
* Multiple Mixer Hamiltonians:
   * `x` and `xy`
* The optimisation loop includes:
   * SciPy Optimisers
   * Custom gradient scypy optimisers

Getting started
================

Installing
------------
Clone the git repository:

.. code-block:: bash
   
   git clone git@github.com:entropicalabs/openqaoa.git

Creating a python `virtual enviorement` for this project is recommended. (for instance, using conda). Instructions on how to create a virtual environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). 
Make sure to use **python 3.8** for the environment.

After cloning the repository `cd openqaoa` and pip install in edit mode:

.. code-block:: bash

   pip install -e .


Your first QAOA workflow
-------------------------
Workflows are a simplified way to run end to end QAOA or RQAOA. In their basic format they consist of the following steps.

First, create a problem instance. For example, an instance of vertex cover:
.. code-block:: python

   from openqaoa.problems.problem import MinimumVertexCover
   import networkx
   g = networkx.circulant_graph(6, [1])
   vc = MinimumVertexCover(g, field =1.0, penalty=10)
   pubo_problem = vc.get_pubo_problem()


Where [networkx](https://networkx.org/) is an open source Python package that can easily, among other things, create graphs.

Once the binary problem is defined, the simplest workflow can be defined as 

.. code-block:: python
   
   from openqaoa.workflows.optimizer import QAOA  
   q = QAOA()
   q.compile(pubo_problem)
   q.optimize()



Workflows can be customised using some convenient setter functions. First, we need to set the device where we want to execute the workflow

.. code-block:: python

   from openqaoa.devices import create_device
   qcs_credentials = {'as_qvm':True, 'execution_timeout' : 10, 'compiler_timeout':10}
   device = create_device(location='qcs',name='6q-qvm',**qcs_credentials)


Then, the QAOA parameters can be set as follow

.. code-block:: python

   q_custom = QAOA()
   q_custom.set_circuit_properties(p=10, param_type='extended', init_type='ramp', mixer_hamiltonian='x')
   q_custom.set_device(device)
   q_custom.set_backend_properties(n_shot=200, cvar_alpha=1)
   q_custom.set_classical_optimizer(method='nelder-mead', maxiter=2)
   q_custom.compile(pubo_problem)
   q_custom.optimize()

Currently, the available devices are:

.. list-table:: Title
   :widths: 25 250
   :header-rows: 1

   * - Device location
     - Device Name
   * - `local`
     - `['qiskit_shot_simulator', 'qiskit_statevec_simulator', 'qiskit_qasm_simulator', 'vectorized', 'pyquil.statevector_simulator']`
   * - `IBMQ`
     - Please check the IMBQ backends available to your account
   * - `Rigetti's QCS`
     - `[nq-qvm, Aspen-11, Aspen-M-1]`

With the notation `nq-qvm` it is intended that `n` is a positive integer. For example, `6q-qvm`.

Your first RQAOA workflow
-------------------------

.. code-block:: python

   from openqaoa.workflows.optimizer import RQAOA
   r = RQAOA(rqaoa_type='adaptive')
   r.set_rqaoa_parameters(n_max=5, n_cutoff = 5)
   r.compile(pubo_problem)
   r.optimize()

rqaoa_type can take two values which select elimination strategies. The user can choose between `adaptive` or `custom`.

Factory mode
------------
The user is also free to directly access the source code without using the workflow API. 

A few reference notebooks can be found:
* [comparing vectorized, pyquil, and qiskit backents](examples/test_backends_correctness.ipynb)
* [Parameter sweep for vectorised](examples/openqaoa_example_vectorised.ipynb)


The basic procedure is the following

First, import all the necessay functions

.. code-block:: python

   from openqaoa.qaoa_parameters import Hamiltonian, QAOACircuitParams, create_qaoa_variational_params
   from openqaoa.utilities import X_mixer_hamiltonian
   from openqaoa.backends.qaoa_backend import QAOAvectorizedBackendSimulator
   from openqaoa.optimizers.qaoa_optimizer import ScipyOptimizer


then, specify terms and weights in order to define the cost hamiltonian

.. code-block:: python

   terms = [(1,2),(2,3),(0,3),(4,0),(1,),(3,)]
   coeffs = [1,2,3,4,3,5]
   n_qubits = 5

   cost_hamil = Hamiltonian.classical_hamiltonian(terms=terms,coeffs=coeffs,constant=0)
   mixer_hamil = X_mixer_hamiltonian(n_qubits=n_qubits)
   After having created the hamiltonians it is time to create the Circuit parameters and the Variational Parameters

.. code-block:: python

   qaoa_circuit_params = QAOACircuitParams(cost_hamil,mixer_hamil,p=1)
   params = create_qaoa_variational_params(qaoa_circuit_params, params_type='fourier',init_type='rand',q=1)

Then proceed by instantiating the backend device

.. code-block:: python
   
   backend_obj = QAOAvectorizedBackendSimulator(circuit_params = qaoa_circuit_params, append_state = None, prepend_state = None, init_hadamard = True)

And finally, create the classical optimizer and minimize the objective function

.. code-block:: python

   optimizer_dict = {'method': 'cobyla', 'maxiter': 10}
   optimizer_obj = ScipyOptimizer(backend_obj, params, optimizer_dict)
   optimizer_obj()


The result of the optimization will the be accessible as 

.. code-block:: python

   optimizer_obj.results_information()


License
========

OpenQAOA is released open source under the Apache License, Version 2.0.

Contents
========

.. toctree::
   :maxdepth: 3
   :caption: About Entropica Labs

   about

.. toctree::
   :maxdepth: 3
   :caption: General reference

   faq
   changelog

.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   notebooks/1_workflows_example.ipynb
   notebooks/2_simulators_comparison.ipynb
   notebooks/3_qaoa_on_qpus.ipynb
   notebooks/4_qaoa_variational_parameters.ipynb
   notebooks/5_advanced_parameterization.ipynb
   notebooks/6_fast_qaoa_simulator.ipynb
   notebooks/7_cost_landscapes_w_manual_mode.ipynb
   notebooks/8_results_example.ipynb
   notebooks/9_RQAOA_example.ipynb


.. toctree::
   :maxdepth: 3
   :caption: API reference

   workflows
   rqaoa
   problems
   qaoaparameters
   backends
   logger_and_results
   optimizers
   utilities


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
