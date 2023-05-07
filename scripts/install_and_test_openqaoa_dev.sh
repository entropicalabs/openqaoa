#Exit immediately if a command exits with a non-zero status.
set -e

# TODO: Dynamically generate names (The order of installing is important here.)
modulesList=("src/openqaoa-core" "src/openqaoa-qiskit" "src/openqaoa-braket" "src/openqaoa-pyquil" "src/openqaoa-azure")

pip install .
pytest tests/ src/*/tests
pip uninstall -y openqaoa