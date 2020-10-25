# TEAM KLR - Project 1 - 2020 Fall Machine Learning (CS-433)

[Full assignment guideline pdf here](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf)


## Organisation

There are three folders : `data`, `src` and `plot`

### Data

In the `data` folder you will find `train.csv`, `test.csv` as well as `sample-submission.csv`.

- `train.csv` is the data used to train our model.
- `test.csv` is the data that we want to classify thanks to our previously trained model.
- `sample-submission.csv` is the predictions on the `test.csv` dataset.

### Plots

In the `plots` folder you will find some important graphs.

- `Comparing MSE and classification loss.pdf`
- `Determining the best degree.pdf`
- `Distant regressand values.pdf`
- `Zoom determining the best degree.pdf`

### Src

In the `src` folder you will find our code. 

- `run.py` is the code to run in order to get the same `sample-submission.csv` as our best submittion on AIcrowd.com.
- `implementations.py` contains multiple functions used in our model. You will find (among other functions that we used for training our model) the `least_squares_GD(y, tx, initial_w, max_iters, gamma)`, `least_squares_SGD(y, tx, initial_w, max_iters, gamma)`, `least_squares(y, tx)`, `ridge_regression(y, tx, lambda_)`, `logistic_regression(y, tx, initial_w, max_iters, gamma)`, `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` required.
- `helpers.py` contains our own useful functions used to handle the data. The provided `proj1_helpers.py` has been left untouched.
- `Visualising Ridge Tuning.ipynb` is the notebook that allowed us to create most of the plots in the folder `plots`, with `select_best_lambda.ipynb` producing the lambda plot. You will find explanations concerning our findings and our thought process in it.

## Usage

Run in the terminal:

```
python run.py
```

Then the result should be in the `data` file in `sample-submission.csv`. 

## Report

Here is our report in pdf format: REPORT.PDF

## Team

- Kamran Nejad-Sattary (kamran.nejad-sattary@epfl.ch)
- Razvan Mocan (razvan-florin.mocan@epfl.ch)
- Loïc Busson (loic.busson@epfl.ch)
