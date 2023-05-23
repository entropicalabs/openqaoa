#Exit immediately if a command exits with a non-zero status.
set -e

pip install .
pytest tests/ src/*/tests -n auto
pip uninstall -y openqaoa