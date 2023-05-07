#Exit immediately if a command exits with a non-zero status.
set -e

# TODO: Dynamically generate names (The order of installing and pushing is important here.)
modulesList=("openqaoa-core" "openqaoa-pyquil")

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

for entry in "${modulesList[@]}"; do
    pip uninstall -y $entry
done