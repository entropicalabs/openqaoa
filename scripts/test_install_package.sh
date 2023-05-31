modulesList=("openqaoa-core" "openqaoa-qiskit" "openqaoa-braket" "openqaoa-pyquil" "openqaoa-azure")

for entry in "${modulesList[@]}"; do
    pip install -i https://test.pypi.org/simple/ --no-deps $entry
done