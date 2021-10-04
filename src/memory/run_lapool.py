import argparse

from tensorflow.keras import Input, Model

from src.layers.lapool import LaPool
from src.memory.training import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--F", type=int, default=1)
args = parser.parse_args()


def make_model(N=None, F=None):
    X_in = Input(shape=(F,), name="X_in")
    A_in = Input(shape=(None,), name="A_in", sparse=True)
    X2, A2 = LaPool(shortest_path_reg=False)([X_in, A_in])

    model = Model([X_in, A_in], [X2])

    return model


run_experiment(F=args.F, method="LaPool", make_model=make_model)
