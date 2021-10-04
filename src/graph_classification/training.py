import numpy as np
import tensorflow as tf
from spektral.datasets import ModelNet, TUDataset
from spektral.transforms import OneHotLabels
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from src.modules.transforms import Float, NormalizeSphere, RemoveEdgeFeats

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def load_dataset(name):
    if name in ModelNet.available_datasets():
        dataset = ModelNet(name, transforms=[NormalizeSphere(), Float()])
        dataset_te = ModelNet(name, test=True, transforms=[NormalizeSphere(), Float()])
        filter_at = 5000
        dataset.filter(lambda g: g.n_nodes <= filter_at)
        dataset_te.filter(lambda g: g.n_nodes <= filter_at)
    elif name in TUDataset.available_datasets():
        dataset = TUDataset(name, transforms=[Float()])
        dataset_te = None
    else:
        raise ValueError(f"Unknown dataset {name}")

    # Remove edge attributes
    dataset.apply(RemoveEdgeFeats())
    if dataset_te is not None:
        dataset_te.apply(RemoveEdgeFeats())

    if dataset.n_labels == 1:
        labels = dataset.map(lambda g: g.y, reduce=np.unique)
        dataset.apply(OneHotLabels(labels=labels))
        if dataset_te is not None:
            dataset_te.apply(OneHotLabels(labels=labels))

    return dataset, dataset_te


def main(
    dataset,
    create_model,
    loader_class,
    learning_rate,
    batch_size,
    patience,
    dataset_te=None,
):
    K.clear_session()
    N_avg = dataset.map(lambda g: g.n_nodes, reduce=lambda res: np.ceil(np.mean(res)))
    ratio = 0.5
    k = int(ratio * N_avg)
    n_out = dataset.n_labels

    if dataset.name == "COLORS-3":
        idx_tr = np.arange(0, 500)
        idx_va = np.arange(500, 2500)
        idx_te = np.arange(2500, len(dataset))
    elif dataset.name == "TRIANGLES":
        idx_tr = np.arange(0, 30000)
        idx_va = np.arange(30000, 35000)
        idx_te = np.arange(35000, len(dataset))
    elif dataset_te is not None:
        l_data = len(dataset)
        idxs = np.random.permutation(l_data)
        idx_tr, idx_va = np.split(idxs, [int(0.8 * l_data)])
        idx_te = None
    else:
        l_data = len(dataset)
        idxs = np.random.permutation(l_data)
        split_va, split_te = int(0.8 * l_data), int(0.9 * l_data)
        idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])

    dataset_tr = dataset[idx_tr]
    dataset_va = dataset[idx_va]
    if idx_te is not None:
        dataset_te = dataset[idx_te]

    # Create loaders
    loader_tr = loader_class(dataset_tr, batch_size=batch_size)
    loader_va = loader_class(dataset_va, batch_size=batch_size)
    loader_te = loader_class(dataset_te, batch_size=batch_size)

    if hasattr(loader_tr, "mask"):
        loader_tr.mask = True
        loader_va.mask = True
        loader_te.mask = True

    # Model
    model = create_model(n_out, k=k, ratio=0.5)

    # Training
    opt = Adam(learning_rate=learning_rate)
    loss_fn = CategoricalCrossentropy()
    acc_fn = CategoricalAccuracy()

    input_signature = loader_tr.tf_signature()

    @tf.function(input_signature=input_signature, experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss_value = loss_fn(target, predictions)  # Main loss
            loss_value += sum(model.losses)  # Auxiliary losses of the model
            acc_value = acc_fn(target, predictions)
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, acc_value

    @tf.function(input_signature=input_signature, experimental_relax_shapes=True)
    def _eval_step(inputs, target):
        predictions = model(inputs, training=False)
        return loss_fn(target, predictions), acc_fn(target, predictions)

    def evaluate(loader):
        output = []
        step = 0
        while step < loader.steps_per_epoch:
            step += 1
            inputs, target = loader.__next__()
            outs = _eval_step(inputs, target)
            output.append(outs)
        return np.mean(output, 0)

    # Training loop
    epoch = step = model_loss = model_acc = best_va_acc = 0
    best_va_loss = np.inf
    best_te_loss = best_te_acc = None
    patience_remain = patience
    es_tol = 1e-6

    for batch in loader_tr:
        outs = train_step(*batch)
        model_loss += outs[0]
        model_acc += outs[1]
        step += 1
        if step == loader_tr.steps_per_epoch:
            epoch += 1
            model_loss /= loader_tr.steps_per_epoch
            model_acc /= loader_tr.steps_per_epoch

            # Compute validation loss and accuracy
            va_loss, va_acc = evaluate(loader_va)

            # Check if loss improved for early stopping
            if va_loss + es_tol < best_va_loss:
                te_loss, te_acc = evaluate(loader_te)
                print(
                    "Epoch {} acc: {:.4f} va_acc: {:.4f} te_acc: {:.4f}".format(
                        epoch, model_acc, va_acc, te_acc
                    )
                )
                best_va_loss = va_loss
                best_va_acc = va_acc
                best_te_loss = te_loss
                best_te_acc = te_acc
                patience_remain = patience
            else:
                patience_remain -= 1
                if patience_remain == 0:
                    break
            model_loss = model_acc = step = 0

    print("Loss: {} - Acc: {}".format(best_te_loss, best_te_acc))
    return best_te_loss, best_te_acc


def run_experiments(
    runs, create_model, loader_class, dataset_name, learning_rate, batch_size, patience
):
    # Data
    dataset, dataset_te = load_dataset(dataset_name)

    results = []
    for r in range(runs):
        print("{} of {}".format(r + 1, runs))
        results.append(
            main(
                dataset,
                create_model,
                loader_class,
                learning_rate,
                batch_size,
                patience,
                dataset_te=dataset_te,
            )
        )
    avg_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    print(
        "{} - Test loss: {:.4f} +- {:.4f} - Test accuracy: {:.4f} +- {:.4f}".format(
            dataset_name, avg_results[0], std_results[0], avg_results[1], std_results[1]
        )
    )

    return avg_results, std_results


def results_to_file(dataset, method, avg_results, std_results):
    filename = "{}_result.csv".format(dataset)
    with open(filename, "a") as f:
        line = "{}, {:.4f} +- {:.4f}, {:.4f} +- {:.4f}\n".format(
            method, avg_results[0], std_results[0], avg_results[1], std_results[1]
        )
        f.write(line)
