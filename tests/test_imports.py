import unittest
import importlib

from setuptools import find_namespace_packages


class TestImports(unittest.TestCase):

    """
    Test module imports within OpenQAOA
    """

    def test_all_module_import(self):
        """
        Test all the main module imports for OQ
        """

        package_names = [
            "openqaoa",
            "openqaoa_braket",
            "openqaoa_qiskit",
            "openqaoa_pyquil",
            "openqaoa_azure",
        ]
        folder_names = [
            "openqaoa-core",
            "openqaoa-braket",
            "openqaoa-qiskit",
            "openqaoa-pyquil",
            "openqaoa-azure",
        ]
        packages_import = find_namespace_packages(where="./src")
        updated_packages = []
        for each_package_name in packages_import:
            for _index, each_folder_name in enumerate(folder_names):
                if each_folder_name in each_package_name:
                    updated_packages.append(
                        each_package_name.replace(
                            each_folder_name, package_names[_index]
                        )
                    )
                    continue

        for each_package in updated_packages:
            try:
                importlib.import_module(each_package)
            except Exception as e:
                raise Exception(e)


if __name__ == "__main__":
    unittest.main()
