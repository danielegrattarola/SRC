import tensorflow as tf
from spektral.layers import ops
from tensorflow.keras import backend as K


def upsampling_from_matrix(inputs):
    if len(inputs) == 4:
        X, A, I, S = inputs
    else:
        X, A, S = inputs
        I = None

    X_out = K.dot(S, X)
    A_out = ops.matmul_at_b_a(ops.transpose(S), A)
    output = [X_out, A_out]

    if I is not None:
        I_out = K.dot(S, K.cast(I[:, None], tf.float32))[:, 0]
        I_out = K.cast(I_out, tf.int32)
        output.append(I_out)

    return output


def upsampling_from_mask(inputs):
    if len(inputs) == 4:
        X, A, I, M = inputs
    else:
        X, A, M = inputs
        I = None

    S = tf.eye(tf.shape(M)[0])
    S = tf.boolean_mask(S, M)
    S = tf.transpose(S)

    if I is not None:
        return upsampling_from_matrix([X, A, I, S])
    else:
        return upsampling_from_matrix([X, A, S])


def upsampling_with_pinv(inputs):
    if len(inputs) == 4:
        X, A, I, S = inputs
    else:
        X, A, S = inputs
        I = None

    S = tf.transpose(tf.linalg.pinv(S))
    return upsampling_from_matrix([X, A, I, S])


def upsampling_with_pinv_from_mask(inputs):
    if len(inputs) == 4:
        X, A, I, M = inputs
    else:
        X, A, M = inputs
        I = None

    S = tf.eye(tf.shape(M)[0])
    S = tf.boolean_mask(S, M)
    S = tf.transpose(S)

    return upsampling_with_pinv([X, A, I, S])
