"""some helper functions."""
import numpy as np
from implementations import compute_loss, compute_gradient_mse, compute_mse
from proj1_helpers import load_csv_data


def standardize(x):
    """
    Standardize data
    :param x matrix(n, d)
    :return x standardized
    """

    # Consider -999 NaN
    inds = np.where(x == -999)
    x[inds] = np.nan

    # Get the median without Nan values for each column
    col_mean = np.nanmedian(x, axis=0)

    # Find indices that you need to replace
    inds = np.where(np.isnan(x))

    # Place column means in the indices. Align the arrays using take
    x[inds] = np.take(col_mean, inds[1])

    # Min max normalization
    xmin, xmax = np.min(x, axis=0), np.max(x, axis=0)
    x = (x - xmin) / (xmax - xmin)

    return x


def load_data(path_dataset, standardize=True, standard_fct=standardize):
    """
    Loads data
    :param path_dataset: data folder containing test.csv and train.csv
    :param standardize should the data be standardized
    :param standard_fct function used for standardization
    :return: y (n,), tX (n,d), ids (n)
    """
    y, tX, ids = load_csv_data(path_dataset)

    if standardize:
        tX = standard_fct(tX)

    return y, tX, ids


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    :param y: (n,) array - labels
    :param tx: (n,d) matrix - inputs
    :param batch_size: int
    :param num_batches: int; number of batches
    :param shuffle: boolean; shuffle the dataset or not to avoid ordering in the original data messing with the
    randomness of the minibatches
    :return: final weights vector and loss
    :return: an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)

    # If !shuffle; easier for processor as branching takes a lot of time
    shuffled_y = y
    shuffled_tx = tx

    if shuffle:
        # If permutation param is an integer, randomly permute ``np.arange(param)``
        shuffle_indices = np.random.permutation(data_size)
        shuffled_y = shuffled_y[shuffle_indices]
        shuffled_tx = shuffled_tx[shuffle_indices]

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        if start_index < end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def lasso_reg(y, tx, initial_w, max_iters, gamma, LAMBDA):
    """
    L1 regularized linear regression = Lasso Regression
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array; initial weights
    :param max_iters: int; number of iterations
    :param gamma: float; learning rate
    :return: final weights vector and loss
    """

    w = initial_w
    for n_iter in range(max_iters):
        # retrieve gradient and cost
        grd, e = compute_gradient_mse(y, tx, w)

        # prepare the regularization factor
        reg = np.sign(w) * (-1)

        # update the weights
        w = w - (grd + reg * LAMBDA) * gamma

        # set small weights to 0
        w[w < 0.15] = 0
        print(f"Step loss: {compute_mse(e)}")

    # calculate the final loss
    loss = compute_loss(y, tx, w, compute_mse) + np.sum(np.abs(w) * LAMBDA)

    return w, loss
