import numpy as np


class Float:
    def __call__(self, graph):
        graph.a = graph.a.astype("f4")
        graph.x = graph.x.astype("f4")

        return graph


class RemoveEdgeFeats:
    def __call__(self, graph):
        graph.e = None

        return graph


class NormalizeSphere:
    def __call__(self, graph):
        offset = np.mean(graph.x, -2, keepdims=True)
        scale = np.abs(graph.x).max()
        graph.x = (graph.x - offset) / scale

        return graph


def normalize_point_cloud(x):
    offset = np.mean(x, -2, keepdims=True)
    scale = np.abs(x).max()
    x = (x - offset) / scale
    x /= np.linalg.norm(x, axis=0, keepdims=True)

    return x
