import numpy as np

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

def compute_mse(y, tx, w):
    
    """
    Calculate the Mean Squared Error for the vector e
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array
    :return: computed cost using mean squared error
    """
    e = y-tx.dot(w)
    return 1/2*np.mean(e**2)

def compute_gradient_mse(y, tx, w):
    
    """
    Compute mse gradient
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array of initial weights
    :return: (d,) array of computed vector
    """
    
    data_size = tx.shape[0]
    e = y - tx.dot(w)
    grd = - tx.T.dot(e) / data_size
    
    return grd

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """
    Gradient descent (MSE) implementation with linear regression
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array of initial weights
    :param max_iters: int indicating maximum iterations
    :param gamma: float indicating learning rate
    :return: final weights vector and loss
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        # retrieve gradient
        grd = compute_gradient_mse(y, tx, w)
        # update step
        w = w - gamma * grd
        
    #calculate the final loss
    loss = compute_mse(y, tx, w)
    
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    """
    Stochastic gradient descent (MSE) implementation with linear regression
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array of initial weights
    :param max_iters: int indicating maximum iterations
    :param gamma: float indicating learning rate
    :return: final weights vector and loss
    """
    
    w = initial_w
    # uniform picking of minibatch of a single datapoint in this case
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            # retrieve gradient
            grd = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            # update step
            w = w - gamma * grd

    #calculate the final loss    
    loss = compute_mse(y, tx, w)
    
    return w, loss

def least_squares(y, tx):
    
    """
    Least squares regression solver using normal equations
    :param y: (n,) array
    :param tx: (n,d) matrix
    :return: optimal weights vector, loss(mse)
    """
    
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    loss = compute_mse(y, tx, w)

    return w, loss

def ridge_regression(y, tx, lambda_):
    
    """
    Ridge regression solver using normal equations
    :param y: (n,) array
    :param tx: (n,d) matrix
    :return: loss(mse), optimal weights vector
    """
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    loss = compute_mse(y, tx, w)

    return w, loss


def compute_sigmoid(xw):

    """
    Computes sigmoid of input vector
    :param xw: (n,) array input to be sigmoid transformed
    :return: (n,) sigmoid-transformed vector
    """

    return 1 / (1 + np.exp(-xw))

def nlog_likelihood(y, tx, w):

    """
    Compute the negative log likelihood loss
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array of weights
    :return: computed loss given by negative log likelihood
    """
    pred = compute_sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1-y).T.dot(np.log(1-pred))
    return np.squeeze(-loss)

def compute_gradient_sigmoid(y, tx, w):

    """
    Compute sigmoid gradient
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array of initial weights
    :return: (d,) array of computed gradient vector
    """

    pred = compute_sigmoid(tx.dot(w))
    grd = tx.T.dot(pred-y)

    return grd

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    """
    Logistic regression using SGD
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array of initial weights
    :param max_iters: int indicating maximum iterations
    :param gamma: float indicating learning rate
    :return: optimal weights vector, loss(mse)
    """

    w = initial_w
    # uniform picking of minibatch of a single datapoint in this case
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            # retrieve gradient
            grd = compute_gradient_sigmoid(minibatch_y, minibatch_tx, w)
            # update step
            w = w - gamma * grd

    loss = nlog_likelihood(y, tx, w)

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    """
    Regularized logistic regression using SGD
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array of initial weights
    :param max_iters: int indicating maximum iterations
    :param gamma: float indicating learning rate
    :return: optimal weights vector, loss(mse)
    """

    w = initial_w
    # uniform picking of minibatch of a single datapoint in this case
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            # retrieve gradient
            grd = compute_gradient_sigmoid(minibatch_y, minibatch_tx, w) + lambda_ * w
            # update step
            w = w - gamma * grd 

    loss = nlog_likelihood(y, tx, w)

    return w, loss
