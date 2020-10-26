import numpy as np
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from implementations import ridge_regression
from helpers import build_poly, build_k_indices, normalize

DATA_PATH = '../data/'
lambda_ = 1e-20
degree = 13
seed = 12
k_fold = 7

# We work with the training data in this notebook
y, x, ids = load_csv_data(DATA_PATH+'train.csv')

x, col_mean, xmin, xmax = normalize(x)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te, y_tr = y[te_indice], y[tr_indice]
    x_te, x_tr = x[te_indice], x[tr_indice]
    
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    
    w, _ = ridge_regression(y_tr, tx_tr, lambda_)
    
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
test_x, _, _, _ = normalize(test_x, col_mean, xmin, xmax)

# create final predictions on testing data and submission csv
y_pred = predict_labels(w, build_poly(test_x, degree))
create_csv_submission(test_ids, y_pred, DATA_PATH+'inferred.csv')
print('Your final submission has been created and is called /data/inferred.csv')
