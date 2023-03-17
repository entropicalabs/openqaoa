# Dynamically generate names (The order of installing and pushing is important here.)
modulesList=("src/openqaoa-core" "src/openqaoa-qiskit" "src/openqaoa-braket" "src/openqaoa-pyquil" "src/openqaoa-azure")

for entry in "${modulesList[@]}"; do
    echo "processing $entry/setup.py"
    cd $entry
    pip install .
    python -m build
    cd "../.."
done