
#!/bin/bash

# Simple script to convert Notebooks from ipynb to rst

echo 'I am running'
# REMOVECELLS=$"--TagRemovePreprocessor.remove_input_tags='{\"hide_input\", \"hide_all\"}' --TagRemovePreprocessor.remove_all_outputs_tags='{\"hide_output\", \"hide_all\"}'"
mkdir ../docs/source/notebooks
cp -r ../examples/* ../docs/source/notebooks/