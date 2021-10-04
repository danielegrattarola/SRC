import numpy as np
import scipy.sparse as sp
from spektral.layers import ops
from tensorflow.keras import backend as K
from tqdm import tqdm


def run_experiment(F, method, make_model):
    last_n = 0
    for n in tqdm(range(10, 10000000, 1000)):
        n = int(n)
        try:
            K.clear_session()
            model = make_model(n, F)
            X = np.random.randn(n, F)
            A = sp.rand(n, n, density=0.2)
            A = A.T + A
            A = ops.sp_matrix_to_sp_tensor(A)
            xout, _ = model([X, A])
            tqdm.write("nout", xout.shape[0])
            last_n = n
        except:
            with open("mem_results.txt", "a") as f:
                f.write(f"{method}\t{F}\t{last_n}\n")
            exit()
