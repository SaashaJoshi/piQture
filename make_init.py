from os.path import isdir, join

root = "."
finit = "__init__.py"


def visitor(arg, dirname, fnames):
    fnames = [fname for fname in fnames if isdir(fname)]
    # here you could do some additional checks ...
    print("adding %s to : %s" % (finit, dirname))
    with open(join(dirname, finit), "w") as file_:
        file_.write("")


walk(root, visitor, None)
