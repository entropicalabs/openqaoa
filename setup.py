from setuptools import setup, find_namespace_packages
from os import getcwd, listdir

current_path = getcwd()

with open("README.md", "r") as fh:
    long_description = fh.read()

# Dev package will share versions with the core in it.
with open("src/openqaoa-core/openqaoa/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

plugin_name = [each_item for each_item in listdir('src') if 'openqaoa-' in each_item]

# Scan plugins for their requirements and collate them.
requirements_dict = {'requirements': [], 
                     'requirements_docs': [], 
                     'requirements_test': []}

for each_key, each_value in requirements_dict.items():
    for each_plugin_name in plugin_name:
        try:
            with open('src/'+each_plugin_name+'/'+each_key+'.txt') as rq_file:
                plugin_requirements = rq_file.readlines()
                plugin_requirements = [r.strip() for r in plugin_requirements]
                requirements_dict[each_key].extend(plugin_requirements)
        except FileNotFoundError:
            continue

requirements = requirements_dict['requirements']
requirements_docs = requirements_dict['requirements_docs']
requirements_test = requirements_dict['requirements_test']

package_names = [
    "openqaoa",
    "openqaoa_braket",
    "openqaoa_qiskit",
    "openqaoa_pyquil",
    "openqaoa_azure",
]
folder_names = [
    "openqaoa-core.openqaoa",
    "openqaoa-braket.openqaoa_braket",
    "openqaoa-qiskit.openqaoa_qiskit",
    "openqaoa-pyquil.openqaoa_pyquil",
    "openqaoa-azure.openqaoa_azure",
]
packages_import = find_namespace_packages(where="./src")
updated_packages = []
for each_package_name in packages_import:
    for _index, each_folder_name in enumerate(folder_names):
        if each_folder_name in each_package_name:
            updated_packages.append(
                each_package_name.replace(each_folder_name, package_names[_index])
            )
            continue

setup(
    name="openqaoa",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
    packages=updated_packages,
    package_dir={
        "": "src",
        "openqaoa": "src/openqaoa-core/openqaoa",
        "openqaoa_braket": "src/openqaoa-braket/openqaoa_braket",
        "openqaoa_qiskit": "src/openqaoa-qiskit/openqaoa_qiskit",
        "openqaoa_pyquil": "src/openqaoa-pyquil/openqaoa_pyquil",
        "openqaoa_azure": "src/openqaoa-azure/openqaoa_azure",
    },
    entry_points={
        "openqaoa.plugins": [
            "qiskit = openqaoa_qiskit.backend_config",
            "braket = openqaoa_braket.backend_config",
            "pyquil = openqaoa_pyquil.backend_config",
            "azure = openqaoa_azure.backend_config"
        ]
    },
    url="https://github.com/entropicalabs/openqaoa",
    install_requires=requirements,
    license="MIT",
    description="OpenQAOA is a python open-source multi-backend Software Development Kit to create, customise and execute the Quantum Approximate Optimisation Algorithm (QAOA) on Noisy Intermediate-Scale Quantum (NISQ) devices, and simulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="quantum optimisation SDK",
    extras_require={
        "docs": requirements_docs,
        "tests": requirements_test,
        "all": requirements_docs + requirements_test,
    },
)
