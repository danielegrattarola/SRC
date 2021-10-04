import tensorflow as tf


def sparse_connect(A, S, N):
    N_sel = tf.cast(tf.shape(S), tf.int64)[0]
    m = tf.scatter_nd(S[:, None], tf.range(N_sel) + 1, (N,)) - 1

    row, col = A.indices[:, 0], A.indices[:, 1]
    r_mask = tf.gather(m, row)
    c_mask = tf.gather(m, col)
    mask_total = (r_mask >= 0) & (c_mask >= 0)
    r_new = tf.boolean_mask(r_mask, mask_total)
    c_new = tf.boolean_mask(c_mask, mask_total)
    v_new = tf.boolean_mask(A.values, mask_total)

    output = tf.SparseTensor(
        values=v_new, indices=tf.stack((r_new, c_new), 1), dense_shape=(N_sel, N_sel)
    )
    return tf.sparse.reorder(output)
