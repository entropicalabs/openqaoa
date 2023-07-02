#Exit immediately if a command exits with a non-zero status.
set -e

# TODO: Dynamically generate names (The order of installing and pushing is important here.)
modulesList=("src/openqaoa-core" "src/openqaoa-qiskit" "src/openqaoa-braket" "src/openqaoa-pyquil" "src/openqaoa-azure")

python scripts/test_version.py
 
pip install build twine

# build and install plugins
for entry in "${modulesList[@]}"; do
    echo "processing $entry/setup.py"
    cd $entry
    rm -rf dist build
    pip install .
    python -m build
    twine upload dist/* --username $1 --password $2
    rm -rf dist build
    cd "../.."
done

# build and install openqaoa metapackage
rm -rf dist build
pip install .
python -m build
twine upload dist/* --username $1 --password $2