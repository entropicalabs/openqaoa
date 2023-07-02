Welcome to OpenQAOA's documentation!
====================================

OpenQAOA is an advanced multi-backend SDK for quantum optimization designed to ease research efforts within the VQA environment while ensuring the reliability and reproducibility of results. The library is divided into individually installable backend plugins. 
The core functionalities of the package are contained within `openqaoa-core`, required to run QAOA computations on any Quantum hardware or simulator. Further it includes `openqaoa-qiskit`, `openqaoa-pyquil`, `openqaoa-azure`, `openqaoa-braket` for running
QAOA on devices accessible through the respective cloud providers. Installing any plugin through PyPI ships `openqaoa-core` along with it to provide the complete set of tools required to run QAOA computations.
Users can also easily install all OpenQAOA plugins available by installing `openqaoa` through PyPI. The `openqaoa` metapackage easily manages all OpenQAOA plugins and their dependencies. Users can also install `openqaoa` in developer mode by git cloning the repository and executing the install Makefile,
instructions for which are provided in the installation section below. This allows users to easily contribute to the OpenQAOA project.


Features
--------

Key features of OpenQAOA:

* Simple yet customizable workflows for QAOA and RQAOA deployable on
   * IBMQ devices
   * Rigetti Quantum Cloud Services
   * Amazon Braket
   * Microsoft Azure Quantum
   * Local simulators (including Rigetti QVM, IBM Qiskit-Aer, and Entropica Labs' vectorized simulator and unit-depth QAOA analytical simulator)
* Multiple parametrization strategies:
   * Standard, Fourier, and Annealing
   * Each class can be further controlled by selecting standard or extended parameter configurations
* Multiple Initliaisation strategies:
   * Linear ramp, random, and custom
* Multiple Mixer Hamiltonians:
   * `x` and `xy`
* The optimization loop includes:
   * SciPy Optimisers
   * Custom gradient SciPy optimizers

Getting started
================

Installing
----------

OpenQAOA provides several installation options to choose from. The package consists of `openqaoa-core` and backend specific modules that let users selectively install the provider they wish to run QAOA on. 
For instance, `openqaoa-qiskit` enables QAOA computations on IBMQ devices and simulators, and qiskit supported devices. For a complete installation including all supported cloud providers,
users can simply install the full `openqaoa` metapackage. Do note, `openqaoa-core` is a dependency for all backend specific modules and the full `openqaoa` pacakge. 
Therefore, it ships by default with all flavors of OpenQAOA installations.

You can install the latest variants of OpenQAOA directly from PyPI. First, we recommend you create a virtual environment with python>=3.10 and then pip install openqaoa variants with the following commands
- To install full OpenQAOA with all backend plugins and the `openqaoa` metapackage, users can

.. code-block:: bash

   pip install openqaoa

- To install specific backend plugin versions of OpenQAOA, users can install `openqaoa-{plugin}`, where `plugin` is populated with the backend of choice

.. code-block:: bash

   pip install openqaoa-plugin

Alternatively, you can install manually directly from the GitHub repository by

1. Clone the git repository:

.. code-block:: bash

   git clone https://github.com/entropicalabs/openqaoa.git

2. We recommend creating a new python `virtual environment`, for instance, using conda. Instructions on how to create a virtual environment using Anaconda can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). Make sure to use **python 3.8** (or newer) for the environment.

3. After cloning the repository `cd openqaoa` and pip install the package. 

.. code-block:: bash

   pip install .

Additionally, users can install OpenQAOA in the developer mode via the Makefile. For a clean editable install of the package run the following command from the `openqaoa` folder.

.. code-block:: bash

   make dev-install

The package can be installed as an editable with extra requirements defined in the `setup.py`. If you would like to install the extra requirements to be able run the tests module or generate the docs, you can run the following

.. code-block:: bash

   make dev-install-x,   with x = {tests, docs, all}

Should you face any issue during the installation, please drop us an email at openqaoa@entropicalabs.com or open an issue!


Your first QAOA workflow
-------------------------
Workflows are a simplified way to run end to end QAOA or RQAOA. In their basic format they consist of the following steps.

First, create a problem instance. For example, an instance of vertex cover:

.. code-block:: python

   from openqaoa.problems import MinimumVertexCover
   import networkx
   g = networkx.circulant_graph(6, [1])
   vc = MinimumVertexCover(g, field =1.0, penalty=10)
   qubo_problem = vc.qubo


Where [networkx](https://networkx.org/) is an open source Python package that can easily, among other things, create graphs.

Once the binary problem is defined, the simplest workflow can be defined as 

.. code-block:: python
   
   from openqaoa import QAOA  
   q = QAOA()
   q.compile(qubo_problem)
   q.optimize()



Workflows can be customised using some convenient setter functions. First, we need to set the device where we want to execute the workflow

.. code-block:: python

   from openqaoa import create_device
   qiskit_sv = create_device(location='local', name='qiskit.statevector_simulator')
   q.set_device(qiskit_sv)

Then, the QAOA parameters can be set as follow

.. code-block:: python

   # circuit properties
   q.set_circuit_properties(p=2, param_type='standard', init_type='rand', mixer_hamiltonian='xy')

   # backend properties (already set by default)
   q.set_backend_properties(prepend_state=None, append_state=None)

   # classical optimizer properties
   q.set_classical_optimizer(method='nelder-mead', maxiter=200,
                           optimization_progress=True, cost_progress=True, parameter_log=True)

Currently, the available devices are:

.. list-table:: Title
   :widths: 25 250
   :header-rows: 1

   * - Device location
     - Device Name
   * - `local`
     - `['qiskit_shot_simulator', 'qiskit_statevec_simulator', 'qiskit_qasm_simulator', 'vectorized', 'pyquil.statevector_simulator']`
   * - `Amazon Braket`
     -  ['IonQ', 'Rigetti', 'OQC']
   * - `IBMQ`
     - Please check the IBMQ backends available to your account
   * - `Rigetti's QCS`
     - `[nq-qvm, Aspen-11, Aspen-M-1]`
   * - `Azure`
     - OpenQAOA supports all gate-based QPU present on Azure Quantum. For the freshest list, please check the [Azure Quantum documentation](https://azure.microsoft.com/en-us/products/quantum/#features)

With the notation `nq-qvm` it is intended that `n` is a positive integer. For example, `6q-qvm`.

Check the [OpenQAOA Website](https://openqaoa.entropicalabs.com/devices/device/) for further details.

Your first RQAOA workflow
-------------------------

.. code-block:: python

   from openqaoa import RQAOA
   r = RQAOA()
   r.set_rqaoa_parameters(rqaoa_type='adaptive', n_max=5, n_cutoff = 5)
   r.compile(qubo_problem)
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

   from openqaoa.qaoa_components import Hamiltonian, QAOADescriptor, create_qaoa_variational_params
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

   qaoa_descriptor = QAOADescriptor(cost_hamil,mixer_hamil,p=1)
   params = create_qaoa_variational_params(qaoa_descriptor, params_type='fourier',init_type='rand',q=1)

Then proceed by instantiating the backend device

.. code-block:: python
   
   backend_obj = QAOAvectorizedBackendSimulator(qaoa_descriptor = qaoa_descriptor, append_state = None, prepend_state = None, init_hadamard = True)

And finally, create the classical optimizer and minimize the objective function

.. code-block:: python

   optimizer_dict = {'method': 'cobyla', 'maxiter': 200}
   optimizer_obj = ScipyOptimizer(backend_obj, params, optimizer_dict)
   optimizer_obj()


The result of the optimization will the be accessible as 

.. code-block:: python

   optimizer_obj.qaoa_result.asdict()


License
========

OpenQAOA is released open source under the MIT license

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
   :caption: Installation and setup

   openqaoa_metapackage_install
   
   openqaoa_core/openqaoa_core_install

   openqaoa_qiskit/openqaoa_qiskit_install

   openqaoa_braket/openqaoa_braket_install

   openqaoa_azure/openqaoa_azure_install

   openqaoa_pyquil/openqaoa_pyquil_install

.. toctree::
   :maxdepth: 3
   :caption: API reference

   openqaoa_core/openqaoa_core_api

   openqaoa_qiskit/qiskit_backends

   openqaoa_braket/braket_backends

   openqaoa_azure/azure_backends

   openqaoa_pyquil/pyquil_backends
   

.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   notebooks/01_workflows_example.ipynb
   notebooks/02_simulators_comparison.ipynb
   notebooks/03_qaoa_on_qpus.ipynb
   notebooks/04_qaoa_variational_parameters.ipynb
   notebooks/05_advanced_parameterization.ipynb
   notebooks/06_fast_qaoa_simulator.ipynb
   notebooks/07_cost_landscapes_w_manual_mode.ipynb
   notebooks/08_results_example.ipynb
   notebooks/09_RQAOA_example.ipynb
   notebooks/10_workflows_on_Amazon_braket.ipynb
   notebooks/11_Mixer_example.ipynb
   notebooks/12_testing_azure.ipynb
   notebooks/13_optimizers.ipynb
   notebooks/14_qaoa_benchmark.ipynb
   notebooks/X_dumping_data.ipynb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
