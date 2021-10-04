import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from spektral.layers import ops
from spektral.utils import laplacian

from src.layers.lapool import laplacian as laplacian_tf
from src.modules.data import make_dataset
from src.modules.logging import logdir
from src.modules.losses import quadratic_loss_tf
from src.modules.transforms import normalize_point_cloud
from src.modules.utils import to_numpy

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def compute_loss(X, A, L, model):
    """
    Evaluate the quadratic loss on the given inputs.
    :param X: node features
    :param A: adjacency matrix
    :param L: laplacian matrix
    :param model: GNN with pooling layer
    :return: scalar loss value
    """
    X_pool, A_pool, _ = model([X, A])
    L_pool = laplacian_tf(A_pool)
    return quadratic_loss_tf(X, X_pool, L, L_pool)


def load_data(name):
    A, X, _ = make_dataset(name, seed=0)
    X = normalize_point_cloud(X)
    L = laplacian(A)
    _, eigvec = np.linalg.eigh(L.toarray())

    # Final graph signal is composed of node features (normalized) and the first 10 eigv
    X = np.concatenate([X, eigvec[:, :10]], axis=-1)

    return X, A, L


def main(X, A, L, create_model, learning_rate, es_patience, es_tol):
    K.clear_session()

    # Create model and set up traning
    model = create_model(k=int(np.ceil(X.shape[-2] / 2)), ratio=0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(X, A):
        with tf.GradientTape() as tape:
            main_loss = compute_loss(X, A, L, model)  # Main loss
            loss_value = main_loss + sum(model.losses)  # Auxiliary losses of the model

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return main_loss

    # Fit model
    patience = es_patience
    best_loss = np.inf
    best_weights = None
    ep = 0
    while True:
        ep += 1
        loss = train_step(X, A)

        if loss + es_tol < best_loss:
            best_loss = loss
            patience = es_patience
            best_weights = model.get_weights()
            print("Epoch {} - New best loss: {:.4e}".format(ep, best_loss))
        else:
            patience -= 1
            if patience == 0:
                break

    model.set_weights(best_weights)
    return model


def run_experiment(
    name, method, create_model, learning_rate, es_patience, es_tol, runs
):
    log_dir = logdir(name)

    # Load data
    X, A, L = load_data(name)

    X = tf.convert_to_tensor(X.astype("f4"))
    A = ops.sp_matrix_to_sp_tensor(A.astype("f4"))
    L = ops.sp_matrix_to_sp_tensor(L.astype("f4"))

    # Run main
    results = []
    for r in range(runs):
        print("{} of {}".format(r + 1, runs))
        model = main(X, A, L, create_model, learning_rate, es_patience, es_tol)
        results.append(compute_loss(X, A, L, model))
    avg_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    # Run trained model to get pooled graph
    X_pool, A_pool, S = model([X, A])

    # Convert selection mask to selection matrix
    S = to_numpy(S)
    if S.ndim == 1:
        S = np.eye(S.shape[0])[:, S.astype(bool)]

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
