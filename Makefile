#This file stores instructions to install openqaoa in developer mode.
# Currently can only install all packages together for developer mode.

.PHONY: dev-install
dev-install:
	pip install -e .
	pip install -e ./openqaoa-core
	pip install -e ./openqaoa-qiskit
	pip install -e ./openqaoa-pyquil
	pip install -e ./openqaoa-braket
	pip install -e ./openqaoa-azure