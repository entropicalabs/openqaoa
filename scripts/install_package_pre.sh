set -e

modulesList=("openqaoa-core" "openqaoa-qiskit" "openqaoa-braket" "openqaoa-pyquil" "openqaoa-azure" "openqaoa")

for entry in "${modulesList[@]}"; do
    pip install --pre $entry
done