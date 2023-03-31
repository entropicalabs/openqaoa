import unittest
import importlib
import os

from setuptools import find_namespace_packages


class TestImports(unittest.TestCase):

    """
    Test module imports within OpenQAOA
    """

    def test_all_module_import(self):
        """
        Test all the main module imports for OQ
        """
        
        folder_names = [each_file for each_file in os.listdir('src') if 'openqaoa-' in each_file]
        
        packages_import = []
        for each_file_name in folder_names:
            packages_import.extend(find_namespace_packages(where="./src/"+each_file_name, exclude=['dist', 'build', 'build.*', 'tests', 'tests.*']))
        print(packages_import)
        packages_import = sorted(packages_import)
        for each_package in packages_import:
            try:
                importlib.import_module(each_package)
            except Exception as e:
                raise Exception(e)


if __name__ == "__main__":
    unittest.main()
