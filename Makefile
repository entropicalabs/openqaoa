#This file stores instructions to install openqaoa in developer mode.
#Currently can only install all packages together for developer mode.

.PHONY: local-install
local-install:
	pip install ./src/openqaoa-core
	pip install ./src/openqaoa-qiskit
	pip install ./src/openqaoa-pyquil
	pip install ./src/openqaoa-braket
	pip install ./src/openqaoa-azure
	pip install .

.PHONY: dev-install
dev-install:
	pip install -e ./src/openqaoa-core
	pip install -e ./src/openqaoa-qiskit
	pip install -e ./src/openqaoa-pyquil
	pip install -e ./src/openqaoa-braket
	pip install -e ./src/openqaoa-azure
	pip install -e .

.PHONY: dev-install-tests
dev-install-tests:
	pip install -e ./src/openqaoa-core[tests]
	pip install -e ./src/openqaoa-qiskit
	pip install -e ./src/openqaoa-pyquil
	pip install -e ./src/openqaoa-braket
	pip install -e ./src/openqaoa-azure
	pip install -e .

.PHONY: dev-install-docs
dev-install-docs:
	pip install -e ./src/openqaoa-core[docs]
	pip install -e ./src/openqaoa-qiskit
	pip install -e ./src/openqaoa-pyquil
	pip install -e ./src/openqaoa-braket
	pip install -e ./src/openqaoa-azure
	pip install -e .

.PHONY: dev-install-all
dev-install-all:
	pip install -e ./src/openqaoa-core[all]
	pip install -e ./src/openqaoa-qiskit
	pip install -e ./src/openqaoa-pyquil
	pip install -e ./src/openqaoa-braket
	pip install -e ./src/openqaoa-azure
	pip install -e .

.PHONY: dev-uninstall
dev-uninstall:
	pip uninstall openqaoa -y
	pip uninstall openqaoa-core -y
	pip uninstall openqaoa-qiskit -y
	pip uninstall openqaoa-pyquil -y
	pip uninstall openqaoa-braket -y
	pip uninstall openqaoa-azure -y