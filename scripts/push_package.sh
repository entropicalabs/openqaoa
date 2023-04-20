# The script will stop if any intermediate file raises an error.
set -e

# TODO: Dynamically generate names (The order of installing and pushing is important here.)
modulesList=("src/openqaoa-core" "src/openqaoa-qiskit" "src/openqaoa-braket" "src/openqaoa-pyquil" "src/openqaoa-azure")

python scripts/test_version.py

for entry in "${modulesList[@]}"; do
    echo "processing $entry/setup.py"
    cd $entry
    pip install .
    # python -m build
    python3 setup.py -q sdist
    python3 setup.py -q bdist_wheel
    twine upload --repository testpypi dist/* --username $1 --password $2
    cd "../.."
done