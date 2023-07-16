#Exit immediately if a command exits with a non-zero status.
set -e

# TODO: Dynamically generate names (The order of installing is important here.)
modulesList=("openqaoa-core" "openqaoa-braket")

for entry in "${modulesList[@]}"; do
    echo "processing src/$entry/setup.py"
    cd src/$entry
    pip install .
    cd "../.."
done

for entry in "${modulesList[@]}"; do
    echo "testing $entry"
    cd src/$entry
    pytest -n auto tests
    cd "../.."
done