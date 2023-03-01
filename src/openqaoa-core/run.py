from setuptools import find_namespace_packages

package_names = [
    "openqaoa"
]
folder_names = [
    "openqaoa-core"
]
packages_import = find_namespace_packages(where=".")
updated_packages = []
for each_package_name in packages_import:
    for _index, each_folder_name in enumerate(folder_names):
        if each_folder_name in each_package_name:
            updated_packages.append(
                each_package_name.replace(each_folder_name, package_names[_index])
            )
            continue
            
print(updated_packages)