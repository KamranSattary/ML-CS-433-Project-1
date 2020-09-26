# Project 1 - 2020 Fall Machine Learning (CS-433) 
## Dataset
[Dataset available here](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs). To load the data, use the same code we used during the labs. You can find an example of a .csv loading function in our provided template code from labs 1 and 2.

### Data file descriptions
train.csv - Training set of 250000 events. The file starts with the ID column, then the label column (the y you have to predict), and finally 30 feature columns.
test.csv - The test set of around 568238 events - Everything as above, except the label is missing.
sample-submission.csv - a sample submission file in the correct format. The sample submission always predicts -1, that is ‘background’.

Zip file containing all 3 above files can be downloaded from the resource section.

For detailed information on the semantics of the features, labels, and weights, see the technical documentation from the LAL website on the task. Note that here for the EPFL course, we use a simpler evaluation metric instead (classification error).

Some details to get started:

- all variables are floating point, except PRI_jet_num which is integer 
- variables prefixed with PRI (for PRImitives) are “raw” quantities about the bunch collision as measured by the detector.
- variables prefixed with DER (for DERived) are quantities computed from the primitive features, which were selected by the physicists of ATLAS.
- it can happen that for some entries some variables are meaningless or cannot be computed; in this case, their value is −999.0, which is outside the normal range of all variables.

## Task
### Methods to implement
We want you to implement and use the methods we have seen in class and in the labs. You will need to provide working implementations of the following functions.
|Function|Details|
|---|---|
|least squares GD(y, tx, initial w,max iters, gamma)|Linear regression using gradient descent| 
|least squares SGD(y, tx, initial w,max iters, gamma)|Linear regression using stochastic gradient descent| 
|least squares(y, tx)|Least squares regression using normal equations| 
|ridge regression(y, tx, lambda )|Ridge regression using normal equations| 
|logistic regression(y, tx, initial w,max iters, gamma)|Logistic regression using gradient descent or SGD| 
|reg logistic regression(y, tx, lambda ,initial w, max iters, gamma)|Regularized logistic regression using gradient descent or SGD| 

In the above method signatures, for iterative methods, initial w is the initial weight vector, gamma is the step-size, and max iters is the number of steps to run. lambda is always the regularization parameter. (Note that here we have used the trailing underscore because lambda is a reserved word in Python with a different meaning). For SGD, you must use the standard mini-batch-size 1 (sample just one datapoint).

You should take care of the following:

- Return type: Note that all functions should return: (w, loss), which is the last weight vector of the method, and the corresponding loss value (cost function). Note that while in previous labs you might have kept track of all encountered w for iterative methods, here we only want the last one.
- File names: Please provide all function implementations in a single python file, called implementations.py.
- All code should be easily readable and commented.
- Note that we might automatically call your provided methods and evaluate for correct implementation

Here are some good practices of scientific computing as a reference: [http://arxiv.org/pdf/1609.00037](http://arxiv.org/pdf/1609.00037) or
an older article [http://arxiv.org/pdf/1210.0530](http://arxiv.org/pdf/1210.0530).

### Submitting Predictions
Once you have a working model (using the above methods or a modified one), you can send your predictions to the competition platform to see how your model is doing against the other teams. You can submit whenever and as many times as you like, until the deadline.

Your predictions must be in .csv format, see sample-submission.csv. You must use the same datapoint ids as in the test set test.csv. To generate .csv output from Python, use our provided helper functions in helpers.py (see project 1 folder on github).

After a submission, aicrowd.com will compute your score on the test set, and will show you your score and ranking in the leaderboard.

This is useful to see how you compare against other teams, but you should not consider this score as the only
evaluation of your model. Always estimate your test error by using a local validation set, or local cross-validation!
This is important to avoid overfitting the test set online. Also, it allows you to make experiments faster, and save
uploading bandwidth :). You are only allowed a maximum of 5 submissions to the submission system per day.

Improving your predictions. While the above described method implementations must be part of your code submission, you can now implement additional modifications of these basic methods above. You can construct better features for the task, or perform better data preprocessing for this particular dataset, or even implement an additional modification of one of the above mentioned ML methods. Note that it is not allowed to use external libraries, code or data in this project. (It will be allowed in Project 2).

## Final submission
Your final submission to the online system (a standard system as used for scientific conferences) must consist of the following:
* Report: Your 2 page report as .pdf
* Code: The complete executable and documented Python code, as one .zip file. Rules for the code part:
  - Reproducibility: In your submission, you must provide a script run.py which produces exactly the same .csv predictions which you used in your best submission to the competition system.
  - Documentation: Your ML system must be clearly described in your PDF report and also well-documented in the code itself. A clear ReadMe file must be provided. The documentation must also include all data preparation, feature generation as well as cross-validation steps that you have used.
  - In addition to your customized system, don’t forget that your code submission must still also include the 6 basic method implementations as described above in step 2.
  - No use of external ML libraries is allowed in Project 1. (It will be allowed in Project 2).
  - No external datasets allowed. \
  
Submission URL: http://mlcourse.epfl.ch
Submission deadline: Oct 26th, 2020 (16:00 afternoon)

## Physics Background
The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles
have mass. Its discovery at the Large Hadron Collider at CERN was announced in March 2013. In this project,
you will apply machine learning techniques to actual CERN particle accelerator data to recreate the process of
“discovering” the Higgs particle. For some background, physicists at CERN smash protons into one another at
high speeds to generate even smaller particles as by-products of the collisions. Rarely, these collisions can produce
a Higgs boson. Since the Higgs boson decays rapidly into other particles, scientists don’t observe it directly,
but rather measure its“decay signature”, or the products that result from its decay process. Since many decay
signatures look similar, it is our job to estimate the likelihood that a given event’s signature was the result of a
Higgs boson (signal) or some other process/particle (background). In practice, this means that you will be given
a vector of features representing the decay signature of a collision event, and asked to predict whether this event
was signal (a Higgs boson) or background (something else). To do this, you will use the binary classification
techniques we have discussed in the lectures.

If you’re interested in more background on this dataset, we point you to the longer description here:
https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf.

Note that understanding the physics background is not necessary to perform well in this machine learning challenge
as part of the course.


