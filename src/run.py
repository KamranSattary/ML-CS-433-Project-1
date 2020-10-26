import numpy as np
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission

DATA_PATH = '../data/'
lambda_ = 1e-20
degree = 13
seed = 12
k_fold = 7

# We work with the training data in this notebook
y, x, ids = load_csv_data(DATA_PATH+'train.csv')

inds = np.where(x == -999)
x[inds] = np.nan

col_mean = np.nanmedian(x, axis=0)
print(col_mean)

#Find indices that you need to replace
inds = np.where(np.isnan(x))

#Place column means in the indices. Align the arrays using take
x[inds] = np.take(col_mean, inds[1])

#Minmax normalization
xmin, xmax = np.min(x, axis=0), np.max(x, axis=0)
x = (x - xmin) / (xmax-xmin)

def build_poly(x, degree):

    """
    Builds polynomial augmented dataset
    :param x: 
    :param degree: 
    :return:
    """
    r = x.copy()
    for deg in range (2,degree+1):
        r = np.c_[r, np.power(x, deg)]
        
    return np.c_[np.ones(r.shape[0]), r]

def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row/k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k*interval: (k+1) * interval] for k in range (k_fold)]
    return np.array(k_indices)

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a,b)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te, y_tr = y[te_indice], y[tr_indice]
    x_te, x_tr = x[te_indice], x[tr_indice]
    
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    
    w = ridge_regression(y_tr, tx_tr, lambda_)
    
    y_tr_pred = predict_labels(w, tx_tr)
    y_te_pred = predict_labels(w, tx_te)
    
    loss_tr = sum(y_tr_pred != y_tr)/len(y_tr)
    loss_te = sum(y_te_pred != y_te)/len(y_te)
    
    return loss_tr, loss_te, w

def cross_validation_ridge():
   
    k_indices = build_k_indices(y, k_fold, seed)

    losses_te = []
    losses_tr = []
    ws = []
     
    for k in range(k_fold):

        loss_tr, loss_te, w = cross_validation(y, x, k_indices, k, lambda_, degree)
        losses_te.append(loss_te)
        losses_tr.append(loss_tr)
        ws.append(w)

    return np.mean(losses_te, axis=0), np.mean(losses_tr, axis=0), np.mean(ws, axis=0)

losses_te, losses_tr, w = cross_validation_ridge()

print(f'Average Missclassification proportion on test folds was {losses_te}. On Train folds it was {losses_tr}.')

test_y, test_x, test_ids = load_csv_data(DATA_PATH + 'test.csv')

# replace missing values with means determined from training data
inds = np.where(test_x==-999)
test_x[inds] = np.take(col_mean, inds[1])

# minmax normalize again with parameters determined by train data
test_x = (test_x - xmin) / (xmax-xmin)

# create final predictions on testing data and submission csv
y_pred = predict_labels(w, build_poly(test_x, degree))
create_csv_submission(test_ids, y_pred, DATA_PATH+'inferred.csv')
