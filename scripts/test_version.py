import os
import toml

version_dict = {}
for each_file in os.listdir("src"):
    if each_file.startswith("openqaoa-"):
        library = (
            each_file.replace("-", "_") if each_file != "openqaoa-core" else "openqaoa"
        )
        version = __import__(library).__version__
        version_dict.update({each_file: version})

version_check = [
    each_item == version_dict["openqaoa-core"] for each_item in version_dict.values()
]

pyproject_data = toml.load("pyproject.toml")
meta_version = pyproject_data["project"]["version"]
version_check.append(meta_version == version_dict["openqaoa-core"])
for each_check in version_check:
    if not each_check:
        raise Exception(
            "All Plugins should have the same version as the core version of OQ. {}".format(
                version_dict
            )
        )
