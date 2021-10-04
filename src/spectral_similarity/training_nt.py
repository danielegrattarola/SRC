import numpy as np
import tensorflow as tf
from spektral.utils import laplacian

from src.modules.logging import logdir
from src.modules.losses import quadratic_loss
from src.modules.utils import to_numpy
# tf.config.run_functions_eagerly(True)
from src.spectral_similarity.training import load_data

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_experiment(name, method, pooling, runs):
    log_dir = logdir(name)

    # Load data
    X, A, L = load_data(name)

    # Run main
    results = []
    for r in range(runs):
        print("{} of {}".format(r + 1, runs))

        # Pooling
        # We run the graph signal through the pooling function so that it adds fake
        # nodes when using Graclus
        np.random.seed(0)
        _, X_new, A_pool, S = pooling(X, A)

        # Convert selection mask to selection matrix
        S = to_numpy(S)
        if S.ndim == 1:
            S = np.eye(S.shape[0])[:, S.astype(bool)]

        X_pool = S.T.dot(X_new)
        L_pool = laplacian(A_pool)
        loss = quadratic_loss(X, X_pool, L, L_pool)
        results.append(loss)
    avg_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    # Save data for plotting
    np.savez(
        log_dir + "{}_{}_matrices.npz".format(method, name),
        X=to_numpy(X),
        A=to_numpy(A),
        X_pool=to_numpy(X_pool),
        A_pool=to_numpy(A_pool),
        S=S,
    )

    return avg_results, std_results


def results_to_file(dataset, method, avg_results, std_results):
    filename = "{}_result.csv".format(dataset)
    with open(filename, "a") as f:
        line = "{}, {:.4f} +- {:.4f}\n".format(method, avg_results, std_results)
        f.write(line)
