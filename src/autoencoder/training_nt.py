import time

import numpy as np
import tensorflow as tf
from spektral.layers import ops
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from src.autoencoder.training import loss_fn
from src.models.autoencoders import Autoencoder
from src.modules.data import make_dataset
from src.modules.logging import logdir
from src.modules.upsampling import upsampling_with_pinv
from src.modules.utils import to_numpy

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def downsampling(inputs):
    X, A, S = inputs
    return ops.modal_dot(S, X, transpose_a=True), ops.matmul_at_b_a(S, A)


def create_model(F):
    pool = Lambda(downsampling)
    lift = Lambda(upsampling_with_pinv)
    model = Autoencoder(F, pool, lift)
    return model


def main(X, A, S, learning_rate, es_patience, es_tol):
    K.clear_session()

    # Build model and set up training
    F = X.shape[-1]
    model = create_model(F)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(model, optimizer, X, A, S):
        with tf.GradientTape() as tape:
            X_pred, _, _, _, _ = model([X, A, S], training=True)
            main_loss = loss_fn(X, X_pred)  # Main loss
            loss_value = main_loss + sum(model.losses)  # Auxiliary losses of the model

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return main_loss

    # Fit model
    patience = es_patience
    best_loss = np.inf
    best_weights = None
    training_times = []
    ep = 0
    while True:
        ep += 1
        timer = time.time()
        loss_out = train_step(model, optimizer, X, A, S)
        training_times.append(time.time() - timer)
        if loss_out + es_tol < best_loss:
            best_loss = loss_out
            patience = es_patience
            best_weights = model.get_weights()
            print("Epoch {} - New best loss: {:.4e}".format(ep, best_loss))
        else:
            patience -= 1
            if patience == 0:
                break

    model.set_weights(best_weights)
    return model, training_times


def run_experiment(name, method, pooling, learning_rate, es_patience, es_tol, runs):
    log_dir = logdir(name)

    # Load data
    A, X, _ = make_dataset(name)

    # Pooling
    A, X, A_pool, S = pooling(X, A)

    X = np.array(X)
    S = to_numpy(S)
    A = ops.sp_matrix_to_sp_tensor(A.astype("f4"))

    # Run main
    results = []
    for r in range(runs):
        print("{} of {}".format(r + 1, runs))
        model, training_times = main(
            X=X,
            A=A,
            S=S,
            learning_rate=learning_rate,
            es_patience=es_patience,
            es_tol=es_tol,
        )

        # Evaluation
        X_pred, _, _, _, _ = model([X, A, S], training=False)
        loss_out = loss_fn(X, X_pred).numpy()
        results.append(loss_out)
        print("Final MSE: {:.4e}".format(loss_out))
    avg_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    # Run trained model to get pooled graph
    X_pred, A_pred, _, X_pool, _ = model([X, A, S], training=False)

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
        X_pred=to_numpy(X_pred),
        A_pred=to_numpy(A_pred),
        S=to_numpy(S),
        loss=loss_out,
        training_times=training_times,
    )

    return avg_results, std_results
