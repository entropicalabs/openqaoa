## Version v0.2.6 (July 10th, 2024)

This version brings two main changes:
* Reintroducing pyquil as part of the main package after fixing a long standing incompatibility with other packages
* Implementation of Zero Noise Extrapolation by @mar-ven , @AdrianoLusso and @victor-onofre

Other minor bugfixes include:
* Figure handling
* Incompatibility of CPLEX with the workflow pipelines
* Naming conventions of rotation angles

## Version v0.2.5 (December 20th, 2023)

This version fixes some breaking changes appearing with qiskit 0.45

## Version v0.2.4 (November 27th, 2023)

This version brings two main changes:
* Temporary removing pyquil from the openqaoa meta package. Pyquil is still installable as a standalone package with `pip install openqaoa-pyquil`, it will not be included in `pip install openqaoa` until the issue has been fixed - https://github.com/entropicalabs/openqaoa/issues/293
  * Installation was made impossible by conflicting dependencies between pyquil and the rest of openqaoa
  * We removed pyquil temporarily
  * We will be updating pyquil and fixing breaking changes in the next release
* New problem class
  * Binary Paint Shop Problem

## Version v0.2.3 (October 25th, 2023)

This version brings the following changes:
* Fix QPU tests due to provider changes
* Fix `backend_qpu`/`device.backend_device` duplicate
* Compatibility of QAOA and RQAOA with Azure Quantum sessions

## Version v0.2.2 (September 20th, 2023)

This version fixes the version of `pyquil` to prevent breaking changes occuring with v4.0.0.

## Version v0.2.1 (September 11th, 2023)

This version fixes an issue detecting the plugins when installing `openqaoa-core` from PyPI.

## Version v0.2.0 (September 8th, 2023)

This new version brings an import change to OpenQAOA, separately installable plugins to access 
different providers.

### What's changed

This version brings the following new features:
* Independently installable plugins
* New problem classes
  * Sherrington-Kirkpatrick (SK)
  * k-Coloring (KColor)
  * TSP in linear programming (TSP_LP)
* Error mitigation technique implementation
  * SPAM twirling
* Improved Github workflows for internal/external release testing
* Additional bug fixes

## Version v0.1.3 (April 21st, 2023)
This version brings a set of new features in OpenQAOA

### What's Changed
* QAOA workflows now support `evaluate_circuit` which lets compute the energy and measurement outcomes for a QAOA circuit for a specified set of parameter values. 
* The library design-wise is moving closer to modularization goal where each hardware provider backend can be installed independently (this feature is expected to fully functional in the next release)
* OpenQAOA v0.1.3 brings in a list of new optimization problems contributed by @alemonbar, including:
   * Maximal Independent Set (MIS)
   * Vehicle Routing Problem (VRP)
   * Bin Packing problem
   * Portfolio Optimization problem

## Version v0.1.2 (March 23rd, 2023)

There was a small bug in the code that resulted in qiskit circuit not compiling, resulting in erros on QPUs
* This bug was fixed
* Two additional unittests were added to test for compilation to avoid these errors in future.

## Version v0.1.1 (February 23rd, 2023)

This release brings the following new features:
* The ability to plug in custom qubit routing solutions for QAOA circuits. 
* AWS managed jobs are now supported through OpenQAOA

### What's Changed

* Refactor
  * The new `GateMapLabel` introduces an updated and a more consistent way to label QAOA gates in the circuit. 
* New Features
  * OpenQAOA now supports specifying custom qubit routing algorithms in an expected format in the QAOA workflow. This is implemented in https://github.com/entropicalabs/openqaoa/pull/179


## Version v0.1.0 (February 17th, 2023)

This release brings major changes to the structure of OpenQAOA. 
This is OpenQAOA's first major release! V0.1.0 contains many new features and quite a few **breaking changes** :)

**notice**: the license has been changed from `apache 2.0` to `MIT`

### What's Changed

* Refactor
  * The code underwent a considerable refactoring effort. The most noticeable change is in the new `openqaoa-core` and other library plugins in the form `openqaoa-xyz`.
  * Refactor of the result objects for RQAOA / QAOA by @raulconchello in https://github.com/entropicalabs/openqaoa/pull/122
* New Features
  * New backend: OpenQAOA is now compatible with Azure by @shahidee44 in https://github.com/entropicalabs/openqaoa/pull/167
  * New circuit Ansatz: now OQ allows for the Alternating Operator Ansatz by @shahidee44 in https://github.com/entropicalabs/openqaoa/pull/85
  * New backend: analytical formula for p=1 by @kidiki in https://github.com/entropicalabs/openqaoa/pull/147
  * Shot Adaptative optimizers by @raulconchello in https://github.com/entropicalabs/openqaoa/pull/123
  * Supporting PennyLane optimizers by @raulconchello in https://github.com/entropicalabs/openqaoa/pull/101
  * JSON dumps methods for RQAOA / QAOA by @raulconchello in https://github.com/entropicalabs/openqaoa/pull/122
* Bug fixes
  * Bugfix: QPU qubit overflow by @shahidee44 in https://github.com/entropicalabs/openqaoa/pull/108
  * fix: Spelling of Oxford Quantum Circuits by @christianbmadsen in https://github.com/entropicalabs/openqaoa/pull/141
  * Bugfix vanishing RQAOA instances after elimination by @kidiki in https://github.com/entropicalabs/openqaoa/pull/158


## Version v0.0.4 (November 14th, 2022)

This release brings improvements to RQAOA workflow and AWS authentication, and a bugfix to TSP problem class.

### What's Changed
* Refactor
  * Authentication Refactor by @shahidee44 in https://github.com/entropicalabs/openqaoa/pull/126
  * RQAOA workflow by @raulconchello in https://github.com/entropicalabs/openqaoa/pull/109
* Fixes
  * Bugfix to Traveling Salesman QUBO Formulation by @Adirlou in https://github.com/entropicalabs/openqaoa/pull/89
  * Use sparse.linalg.expm for exponentiation of sparse matrices (for sci… by @shaohenc in https://github.com/entropicalabs/openqaoa/pull/121
* Docs
  * Fixing the docs by @Q-lds in https://github.com/entropicalabs/openqaoa/pull/113
  * couple of cosmetic fixes to docs by @vishal-ph in https://github.com/entropicalabs/openqaoa/pull/128

## New Contributors
* @Adirlou made their first contribution in https://github.com/entropicalabs/openqaoa/pull/89

**Full Changelog**: https://github.com/entropicalabs/openqaoa/compare/v0.0.3...v0.0.4

## Version v0.0.3 (October 29th, 2022)

A release to fix two QPU-related bugs

### What's Changed
* Fixes
  * Now save_intermediate works for backends with no jobid #110 
  * Fix a bug that prevented the correct usage of Rigetti's QPUs when on QCS #116 
* Docs
  * Add community examples containing:
    * MVP bipartite graph examples #104 
    * Docplex and general application tutorials #90  

### New Contributors
* @MaldoAlberto made their first contribution in https://github.com/entropicalabs/openqaoa/pull/90
* @krankgeflugel made their first contribution in https://github.com/entropicalabs/openqaoa/pull/104

## Version v0.0.2 (October 19th, 2022)
* Dev
  * OpenQAOA integrates with Amazon Braket!
  * New plotting functions to visualise the most probable bit strings 
  * Introduced a `lowest_cost_bitstrings` function
  * Now it is possible to write a problem statement in DOcplex and have it automatically converted into an OpenQAOA `QUBO`
  * Unbalanced penalisation strategy for slack variables
* Code
  * New FAQ in the documentation
  * Distinct GitHub actions for dev and main branch testing
  * GitHub actions for publishing
  * GitHub actions for code quality and security
* Fixes
  * `qiskit` Backend seed simulator fix
  * CVaR bug fix
  * Fix `knapsack` generation bug
  * Gatemap bug fix

[OpenQAOA reference paper](https://arxiv.org/abs/2210.08695) uploaded to the arxiv 
 
### New Contributors
* @alejomonbar made their first contribution in https://github.com/entropicalabs/openqaoa/pull/72
* @raulconchello made their first contribution in https://github.com/entropicalabs/openqaoa/pull/86
* @EmilianoG-byte made their first contribution in https://github.com/entropicalabs/openqaoa/pull/82

## Version v0.0.1 (October 2nd, 2022)

- **Initial public release**: on [Github][Github] and [PyPI][PyPI].

## Version v0.0.1-beta (July 13th, 2022)

- **Initial release (internal).**

[Github]: https://github.com/entropicalabs/openqaoa
[PyPI]: https://pypi.org/project/openqaoa/
