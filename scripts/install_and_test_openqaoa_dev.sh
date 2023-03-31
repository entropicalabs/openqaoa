# The script will stop if any intermediate file raises an error.
set -e

# TODO: Dynamically generate names (The order of installing is important here.)
modulesList=("src/openqaoa-core" "src/openqaoa-qiskit" "src/openqaoa-braket" "src/openqaoa-pyquil" "src/openqaoa-azure")

pip install .
pytest tests $1 $2 $3 $4 $5

for entry in "${modulesList[@]}"; do
    echo "testing $entry"
    cd $entry
    pytest -n auto tests $1 $2 $3 $4 $5
    cd "../.."
done

pip uninstall -y openqaoa