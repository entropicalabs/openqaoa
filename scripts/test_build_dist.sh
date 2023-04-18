# The script will stop if any intermediate file raises an error.
set -e

# TODO: Dynamically generate names (The order of installing and pushing is important here.)
modulesList=("src/openqaoa-core" "src/openqaoa-qiskit" "src/openqaoa-braket" "src/openqaoa-pyquil" "src/openqaoa-azure")

# This script checks that all internal plugins have the same version number
python scripts/test_version.py

for entry in "${modulesList[@]}"; do
    echo "processing $entry/setup.py"
    cd $entry
    pip install -e .
    python -m build
    cd "../.."
done