modulesList=("openqaoa-core" "openqaoa-qiskit" "openqaoa-braket" "openqaoa-pyquil" "openqaoa-azure")

for entry in "${modulesList[@]}"; do
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --pre $entry
done