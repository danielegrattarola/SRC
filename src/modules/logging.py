import os


def logdir(name):
    path = "./{}/".format(name)
    if not os.path.exists(path):
        os.makedirs(path)

    return path
