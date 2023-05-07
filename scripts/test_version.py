import os
import importlib

version_dict = {}
for each_file in os.listdir("src"):
    if each_file.startswith("openqaoa-"):
        __version__ = ""
        if not each_file == "openqaoa-core":
            exec(
                open(
                    "./src/"
                    + each_file
                    + "/"
                    + each_file.replace("-", "_")
                    + "/_version.py"
                ).read()
            )
        else:
            exec(open("./src/" + each_file + "/openqaoa/_version.py").read())
        version_dict.update({each_file: __version__})

version_check = [
    each_item == version_dict["openqaoa-core"] for each_item in version_dict.values()
]
for each_check in version_check:
    if not each_check:
        raise Exception(
            "All Plugins should have the same version as the core version of OQ. {}".format(
                version_dict
            )
        )
