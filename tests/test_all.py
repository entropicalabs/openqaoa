import os

for pathfile in os.listdir():
    print(pathfile)
    if pathfile != "test_all.py" and pathfile[-3:] == ".py":
        os.system("python3 {} -b".format(pathfile))
