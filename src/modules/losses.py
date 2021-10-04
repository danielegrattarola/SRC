import numpy as np
import tensorflow as tf
from spektral.layers import ops


def quadratic_loss(x, x_pool, L, L_pool):
    if x.ndim == 1:
        x = x[:, None]
        x_pool = x_pool[:, None]
    loss = np.abs(np.dot(x.T, L.dot(x)) - np.dot(x_pool.T, L_pool.dot(x_pool)))
    return np.mean(np.diag(loss))


def quadratic_loss_tf(x, x_pool, L, L_pool):
    if len(x.shape) == 1:
        x = x[:, None]
        x_pool = x_pool[:, None]
    loss = tf.abs(
        tf.matmul(tf.transpose(x), ops.dot(L, x))
        - tf.matmul(tf.transpose(x_pool), ops.dot(L_pool, x_pool))
    )
    return tf.reduce_mean(tf.linalg.diag_part(loss))
