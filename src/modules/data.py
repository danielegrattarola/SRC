import os

import networkx as nx
import numpy as np
from pygsp import graphs
from spektral.datasets import Citation
from spektral.utils import load_off

from src.modules.transforms import normalize_point_cloud

MODELNET_CONFIG = {
    "Airplane": {
        "classname": "airplane",
        "split": "train",
        "sample": 151,
    },
    "Car": {
        "classname": "car",
        "split": "train",
        "sample": 79,
    },
    "Guitar": {
        "classname": "guitar",
        "split": "train",
        "sample": 38,
    },
    "Person": {
        "classname": "person",
        "split": "train",
        "sample": 83,
    },
}


def make_dataset(name, **kwargs):
    if "seed" in kwargs:
        np.random.seed(kwargs.pop("seed"))
    if name in graphs.__all__:
        return make_cloud(name)
    if name in MODELNET_CONFIG:
        return make_modelnet(**MODELNET_CONFIG[name])
    if name in Citation.available_datasets():
        return make_citation(name)


def make_cloud(name):
    if name.lower() == "grid2d":
        G = graphs.Grid2d(N1=8, N2=8)
    elif name.lower() == "ring":
        G = graphs.Ring(N=64)
    elif name.lower() == "bunny":
        G = graphs.Bunny()
    elif name.lower() == "airfoil":
        G = graphs.Airfoil()
    elif name.lower() == "minnesota":
        G = graphs.Minnesota()
    elif name.lower() == "sensor":
        G = graphs.Sensor(N=64)
    elif name.lower() == "community":
        G = graphs.Community(N=64)
    elif name.lower() == "barabasialbert":
        G = graphs.BarabasiAlbert(N=64)
    elif name.lower() == "davidsensornet":
        G = graphs.DavidSensorNet(N=64)
    elif name.lower() == "erdosrenyi":
        G = graphs.ErdosRenyi(N=64)
    else:
        raise ValueError("Unknown dataset: {}".format(name))

    if not hasattr(G, "coords"):
        G.set_coordinates(kind="spring")
    x = G.coords.astype(np.float32)
    y = np.zeros(x.shape[0])  # X[:,0] + X[:,1]
    A = G.W
    if A.dtype.kind == "b":
        A = A.astype("i")

    return A, x, y


def make_modelnet(classname="airplane", split="train", sample=151):
    path = os.path.expanduser(
        f"~/.spektral/datasets/ModelNet/40/{classname}/{split}/{classname}_{sample:04d}.off"
    )
    graph = load_off(path)
    x, a = graph.x, graph.a
    x = normalize_point_cloud(x)

    return a, x, classname


def make_citation(name):
    graph = Citation(name)[0]
    x, a = graph.x, graph.a

    gg = nx.Graph(a)
    lay = nx.spring_layout(gg)
    x = np.array([lay[i] for i in range(a.shape[0])])

    return a, x, name
