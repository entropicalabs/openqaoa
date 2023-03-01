from setuptools import setup, find_namespace_packages
from os import getcwd

current_path = getcwd()

# with open("README.md", "r") as fh:
#     long_description = fh.read()

long_description = ""

with open("_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "openqaoa-core=={}".format(version),
    "openqaoa-qiskit=={}".format(version),
    "azure-quantum",
    "qdk",
    "qiskit-qir",
    "qiskit-ionq",
    "azure-quantum[qiskit]",
]

# requirements_docs = [
#     "sphinx>=4.5.0",
#     "sphinx-autodoc-typehints>=1.18.1",
#     "sphinx-rtd-theme>=1.0.0",
#     "nbsphinx>=0.8.9",
#     "ipython>=8.10.0",
#     "nbconvert>=6.5.1",
# ]

# requirements_test = [
#     "pytest>=7.1.0",
#     "pytest-cov>=3.0.0",
#     "ipython>=8.2.0",
#     "nbconvert>=6.5.1",
#     "pandas>=1.4.3",
#     "plotly>=5.9.0",
#     "cplex>=22.1.0.0",
# ]

# package_names = [
#     "openqaoa",
#     "openqaoa_braket",
#     "openqaoa_qiskit",
#     "openqaoa_pyquil",
#     "openqaoa_azure",
# ]
# folder_names = [
#     "openqaoa-core",
#     "openqaoa-braket",
#     "openqaoa-qiskit",
#     "openqaoa-pyquil",
#     "openqaoa-azure",
# ]
# packages_import = find_namespace_packages(where="./src")
# updated_packages = []
# for each_package_name in packages_import:
#     for _index, each_folder_name in enumerate(folder_names):
#         if each_folder_name in each_package_name:
#             updated_packages.append(
#                 each_package_name.replace(each_folder_name, package_names[_index])
#             )
#             continue

package_name = 'openqaoa_azure'
packages_import = find_namespace_packages(where=".")
updated_packages = [package_name] + [package_name+each_package for each_package in packages_import]

setup(
    name="openqaoa-azure",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
    packages=updated_packages,
    url="https://github.com/entropicalabs/openqaoa",
    install_requires=requirements,
    license="MIT",
    description="Azure Plug-in for OpenQAOA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="quantum optimisation SDK",
)
