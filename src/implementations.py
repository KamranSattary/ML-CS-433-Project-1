def compute_mse(e):
    """
    Calculate the Mean Squared Error for the vector e
    :param e: (n,) array consisting of the error term
    :return: computed cost using mean squared error
    """
    return 1/2*np.mean(e**2)

def compute_mae(e):
    """
    Calculate the Mean Absolute Error for the vector e
    :param e: (n,) array consisting of the error term
    :return: computed cost using mean absolute error
    """
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, loss_function=compute_mse):
    """
    Wrapper for the mse & mae computations, calculate e
    and then use either mse or mae
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array
    :param loss_function: function to use to compute the loss, compute_mse and compute_mae currently supported
    :return: computed cost using mean squared error or mean absolute error
    """
    e = y-tx.dot(w)
    return loss_function(y-tx @ w)

def compute_gradient_mse(y, tx, w):
    """
    Compute mse gradient
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array of initial weights
    :return: (d,) array of computed gradient
    """
    data_size = len(y)
    e = y - tx @ w
    grd = - tx.T @ e / data_size
    return grd, e

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent (MSE) implementation with linear regression
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array of initial weights
    :param max_iters: int indicating maximum iterations
    :param gamma: float indicating learning rate
    :return: lists of losses and weights determined at each step
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # retrieve gradient and cost
        grd, e = compute_gradient_mse(y, tx, w)
        loss = compute_mse(e)
        # update step
        w = w - grd * gamma
        # document step
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent (MSE) implementation with linear regression
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array of initial weights
    :param max_iters: int indicating maximum iterations
    :param gamma: float indicating learning rate
    :return: lists of losses and weights determined at each step
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    # uniform picking of minibatch of a single datapoint in this case
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
        # retrieve gradient and cost
        grd, e = compute_gradient_mse(minibatch_y, minibatch_tx, w)
        loss = compute_mse(e)
        # update step
        w = w - grd * gamma
        # document step
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws