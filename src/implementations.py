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
    Wrapper for the mse & mae cost computations, calculates e and then returns either mse (default) or mae
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array
    :param loss_function: function to use to compute the loss, compute_mse (default) and compute_mae currently supported
    :return: computed cost using mean squared error or mean absolute error
    """
    
    return loss_function(y - tx @ w)

def compute_gradient_mse(y, tx, w):
    
    """
    Compute mse gradient
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array of initial weights
    :return: (d,) array of computed vector
    """
    
    data_size = tx.shape[0]
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
    :return: final weights vector and loss
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        # retrieve gradient and cost
        grd, e = compute_gradient_mse(y, tx, w)
        # update step
        w = w - grd * gamma
        print(f"Step loss: {compute_mse(e)}")
        
    #calculate the final loss
    loss = compute_loss(y, tx, w, compute_mse)
    
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
            # retrieve gradient and cost
            grd, e = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            # update step
            w = w - grd * gamma
            print(f"Step loss: {compute_mse(e)}")

    #calculate the final loss    
    loss = compute_loss(y, tx, w, compute_mse)
    
    return w, loss

def least_squares(y, tx):
    
    """
    Least squares regression solver using normal equations
    :param y: (n,) array
    :param tx: (n,d) matrix
    :return: optimal weights vector, loss(mse)
    """
    
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A,b)
    loss = compute_loss(y, tx, w, compute_mse)

    return w, loss

def ridge_regression(y, tx, lambda_):
    
    """
    Ridge regression solver using normal equations
    :param y: (n,) array
    :param tx: (n,d) matrix
    :return: loss(mse), optimal weights vector
    """
    
    A = tx.T @ tx + (tx.shape[0] * 2 * lambda_ * np.eye(tx.shape[1]))
    b = tx.T @ y
    w = np.linalg.solve(A,b)
    loss = compute_loss(y, tx, w, compute_mse)

    return w, loss


def compute_sigmoid(xw):

    """
    Computes sigmoid of input vector
    :param xw: (n,) array input to be sigmoid transformed
    :return: (n,) sigmoid-transformed vector
    """

    return 1 / (1 + np.exp(-xw))

def log_likelihood(y, tx, w):

    """
    Compute the negative log likelihood loss
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array of weights
    :return: computed loss given by negative log likelihood
    """
    xw = tx.T @ w
    return np.sum(np.log(1 + np.exp(xw)) -  y @ xw)

def compute_gradient_sigmoid(y, tx, w):

    """
    Compute sigmoid gradient
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param w: (d,) array of initial weights
    :return: (d,) array of computed gradient vector
    """

    e = compute_sigmoid(tx @ w) - y
    grd = tx.T @ e

    return grd, e

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
            # retrieve gradient and cost
            grd = compute_gradient_sigmoid(minibatch_y, minibatch_tx, w)
            # update step
            w = w - grd * gamma
            loss = log_likelihood(y, tx, w)
            print(f"Step loss: {loss}")

    return w, loss

# needs to be adapted for regularization, maybe add a parameter to the compute gradient with
# penalty = 'l1' or None options
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

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
            # retrieve gradient and cost
            grd, e = compute_gradient_sigmoid(minibatch_y, minibatch_tx, w)
            # update step
            w = w - grd * gamma
            loss = log_likelihood(y, tx, w)
            print(f"Step loss: {loss}")

    return w, loss

