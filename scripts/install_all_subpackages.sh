#Exit immediately if a command exits with a non-zero status.
set -e

# The order of install is important therefore the list cannot be dynamically generated
modulesList=("openqaoa-core" "openqaoa-qiskit" "openqaoa-braket" "openqaoa-pyquil" "openqaoa-azure")

for entry in "${modulesList[@]}"; do
    echo "processing src/$entry/setup.py"
    cd src/$entry
    pip install .
    cd "../.."
done