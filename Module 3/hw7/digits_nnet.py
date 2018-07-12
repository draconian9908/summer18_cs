#
# digits5: modeling the digits data with DTs and RFs
#


import numpy as np
import pandas as pd

from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

#
# The "answers" to the 20 unknown digits, labeled -1:
#
answers = [9,9,5,5,6,5,0,9,8,9,8,4,0,1,2,3,4,5,6,7]


print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('digits.csv', header=0)    # read the file w/header row #0
df.head()                                 # first five lines
df.info()                                 # column details
print("\n+++ End of pandas +++\n")


print("+++ Start of numpy/scikit-learn +++\n")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_all = df[ '64' ].values      # individually addressable columns (by name)

X_unknown = X_all[:22,:]
y_unknown = y_all[:22]

X_known = X_all[22:,:]
y_known = y_all[22:]

#
# we can scramble the remaining data if we want to (we do)
#
KNOWN_SIZE = len(y_known)
indices = np.random.permutation(KNOWN_SIZE)  # this scrambles the data each time
X_known = X_known[indices]
y_known = y_known[indices]

#
# from the known data, create training and testing datasets
#
TRAIN_FRACTION = 0.85
TRAIN_SIZE = int(TRAIN_FRACTION*KNOWN_SIZE)
TEST_SIZE = KNOWN_SIZE - TRAIN_SIZE   # not really needed, but...
X_train = X_known[:TRAIN_SIZE]
y_train = y_known[:TRAIN_SIZE]

X_test = X_known[TRAIN_SIZE:]
y_test = y_known[TRAIN_SIZE:]

USE_SCALER = False
if USE_SCALER == True:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)   # Fit only to the training dataframe
    # now, rescale inputs -- both testing and training
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_unknown = scaler.transform(X_unknown)

mlp = MLPClassifier(hidden_layer_sizes=(300,200,100,), max_iter=200, alpha=1e-4,
                    solver='sgd', verbose=True, shuffle=True, early_stopping = False, # tol=1e-4,
                    random_state=None, # reproduceability
                    learning_rate_init=.01, learning_rate = 'adaptive')

print("\n\n++++++++++  TRAINING  +++++++++++++++\n\n")
mlp.fit(X_train, y_train)
print("\n++++++++++  TRAINING COMPLETE  +++++++++++++++\n\n")

print("\n++++++++++++  TESTING  +++++++++++++\n\n")
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# predictions:
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("\nConfusion matrix:")
print(confusion_matrix(y_test,predictions))

print("\nClassification report")
print(classification_report(y_test,predictions))
print("\n++++++++++++  TESTING COMPLETE  +++++++++++++\n\n")


print("\n++++++++++++  CLASSIFYING  +++++++++++++\n\n")
# unknown data rows...
#
correct_values = [0,0,0,1,7,2,3,4,0,1,9,9,5,5,6,5,0,9,8,9,8,4]
unknown_predictions = mlp.predict(X_unknown)
print("Unknown predictions:")
print("  Correct values:   [0 0 0 1 7 2 3 4 0 1 9 9 5 5 6 5 0 9 8 9 8 4]")
print("  Our predictions: ", unknown_predictions)

correct = 0
for i in range(len(correct_values)):
    # print("unknown:", unknown_predictions[i], "\nknown:", correct_values[i])
    if unknown_predictions[i] == correct_values[i]:
        correct += 1
accuracy = (correct/len(correct_values)) * 100
print("\nPercent Accuracy:    " + str(accuracy) + "%")
