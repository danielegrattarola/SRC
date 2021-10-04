import numpy as np
import scipy.sparse as sp
from spektral.data import Graph


def read_off(fname):
    lines = open(fname, "r").read().lstrip("OF\n").splitlines()
    x, faces = parse_off(lines)
    n = x.shape[0]
    row, col = np.vstack((faces[:, :2], faces[:, 1:], faces[:, ::2])).T
    adj = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(n, n)).tolil()
    adj[col, row] = adj[row, col]
    adj = adj.T.tocsr()

    return Graph(x=x, adj=adj)


def parse_off(lines):
    n_verts, n_faces, _ = map(int, lines[0].split(" "))

    # Read vertices
    verts = np.array([l.split(" ") for l in lines[1 : n_verts + 1]]).astype(float)

    # Read faces
    faces = lines[n_verts + 1 : n_verts + 1 + n_faces]
    faces = [list(map(int, f.split(" "))) for f in faces]
    triangles = np.array(list(filter(lambda f: len(f) == 4, faces))).astype(int)
    rectangles = np.array(list(filter(lambda f: len(f) == 5, faces))).astype(int)
    if len(rectangles) > 0:
        tri_a = rectangles[:, [1, 2, 3]]
        tri_b = rectangles[:, [1, 2, 4]]
        triangles = np.vstack((triangles, tri_a, tri_b))
    triangles = triangles[:, 1:]
    triangles = triangles[triangles[:, 0].argsort()]

    return verts, triangles
