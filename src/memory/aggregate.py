import argparse
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--method", default=None)
parser.add_argument("--N", default=None)
args = parser.parse_args()

if args.method is not None:
    method = args.method.lower()
    files = glob("{}_*.txt".format(method))
    for f in files:
        d = np.loadtxt(f)
        n = f.split("_")[-1][:-4]
        plt.plot(d, label=n)
    plt.show()

if args.N is not None:
    files = glob("*_*.txt".format(args))
    Ns = sorted(list(set([int(f.split("_")[-1][:-4]) for f in files])))
    methods = set([f.split("_")[0] for f in files])
    values = {m: [] for m in methods}
    for n in Ns:
        for m in methods:
            try:
                d = np.loadtxt(f"{m}_{n}.txt")
            except:
                d = np.zeros(1)
            values[m].append(d.max() - d.min())
    Ns = list(map(str, Ns))
    import pandas as pd

    df = pd.DataFrame({m: values[m] for m in methods}, index=Ns)
    df.plot.bar()
    plt.savefig("memory.pdf")
