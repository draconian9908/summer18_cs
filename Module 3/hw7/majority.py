

# !conda update scikit-learn

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd


#
# learn a two-input, two-output classifier
#
# AND:
# OR:
# XOR: ???
#



#
# try an iris classifier...
#


print("+++ Start of MAJORITY example +++\n")

X_data_complete = np.array( [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]] )
y_data_complete = np.array( [0,1,1,1,0,0,0,1] )

X_unknown = X_data_complete
y_unknown = y_data_complete

X_known = X_data_complete
y_known = y_data_complete

#
# we can scramble the remaining data if we want to (we do)
#
KNOWN_SIZE = len(y_known)
indices = np.random.permutation(KNOWN_SIZE)  # this scrambles the data each time
X_known = X_known[indices]
y_known = y_known[indices]

#
# from the known data, create training and testing datasets
# Here, training == testing == everything!
#
X_train = X_known
y_train = y_known

X_test = X_known
y_test = y_known

#
# it's important to keep the input values in the 0-to-1 or -1-to-1 range
#    This is done through the "StandardScaler" in scikit-learn
#
USE_SCALER = False
if USE_SCALER == True:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)   # Fit only to the training dataframe
    # now, rescale inputs -- both testing and training
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_unknown = scaler.transform(X_unknown)

# scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(3,5,7,11,25,42,), max_iter=200, # 25,11,9,7,5,3,
                    solver='sgd', verbose=True, shuffle=True,
                    early_stopping = False, # tol=1e-4,
                    random_state=None, # reproduceability!
                    learning_rate_init=.1, learning_rate = 'adaptive')

print("\n\n++++++++++  TRAINING  +++++++++++++++\n\n")
mlp.fit(X_train, y_train)


print("\n\n++++++++++++  TESTING  +++++++++++++\n\n")
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# let's see the coefficients -- the nnet weights!
# CS = [coef.shape for coef in mlp.coefs_]
# print(CS)

# predictions:
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("\nConfusion matrix:")
print(confusion_matrix(y_test,predictions))

print("\nClassification report")
print(classification_report(y_test,predictions))


# unknown data rows...
#
unknown_predictions = mlp.predict(X_unknown)
print("Unknown predictions:")
print("  Correct values:   [0 0 0 0 1 1 1 1]")
print("  Our predictions: ", unknown_predictions)


if False:
    L = [5.2, 4.1, 1.5, 0.1]
    row = np.array(L)  # makes an array-row
    row = row.reshape(1,4)   # makes an array of array-row
    if USE_SCALER == True:
        row = scaler.transform(row)
    print("\nrow is", row)
    print("mlp.predict_proba(row) == ", mlp.predict_proba(row))

# C = R.reshape(-1,1)  # make a column!
