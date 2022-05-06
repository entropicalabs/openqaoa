.. OpenQAOA documentation master file, created by
   sphinx-quickstart on Fri Apr 29 05:09:06 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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
Make sure to use **python 3.8.8** for the environment.

After cloning the repository `cd openqaoa` and pip install in edit mode:

.. code-block:: bash

   pip install -e .


Your first QAOA workflow
-------------------------
Workflows are a simplified way to run end to end QAOA or RQAOA. In their basic format they consist of the following steps.

A reference jupyter notebook can be found [here](Workflows_example.ipynb)

.. code-block:: python
   
   from openqaoa.workflows.optimizer import QAOA  
   q = QAOA()
   q.compile(pubo_problem)
   q.optimize()


where the `pubo_problem` is any instance from `openqaoa.problems.problem`. For example, `pubo_problem = NumberPartition([1,2,3]).get_pubo_problem()`.

Workflows can be customised using some convenient setter functions

.. code-block:: python

   q_custom = QAOA()
   q_custom.set_circuit_properties(p=10, param_type='extended', init_type='ramp', mixer_hamiltonian='x')
   q_custom.set_device_properties(device_location='qcs', device_name='Aspen-11')
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
   * - `locale`
     - `['qiskit_shot_simulator', 'qiskit_statevec_simulator', 'qiskit_qasm_simulator', 'vectorized', 'nq-qvm', 'Aspen-11']`
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

License
========

OpenQAOA is released open source under the Apache License, Version 2.0.

Contents
========

.. toctree::
   :maxdepth: 3
   :caption: Workflows

   workflows


.. toctree::
   :maxdepth: 3
   :caption: Inputs

   problems

.. toctree::
   :maxdepth: 3
   :caption: Parametrisation

   qaoaparameters

.. toctree::
   :maxdepth: 3
   :caption: Backend and devices

   backends

.. toctree::
   :maxdepth: 3
   :caption: Classical Optimisers

   optimizers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
