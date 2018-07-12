#
# read iris data
#

import numpy as np
import pandas as pd

from sklearn import tree      # for decision trees
from sklearn import ensemble  # for random forests

try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")


#
# Here are the correct answers to the csv's "unknown" flowers
#
answers = [ 'virginica',   # index 0 (row 1 in the csv)
            'virginica',   # index 1 (row 2 in the csv)
            'versicolor',  # and so on...
            'versicolor',
            'setosa',
            'setosa',
            'virginica',
            'versicolor',
            'setosa']



print("+++ Start of pandas' datahandling +++\n")

# df is a "dataframe":
df = pd.read_csv('iris5.csv', header=0)   # read the file w/header row #0

# Now, let's take a look at a bit of the dataframe, df:
df.head()                                 # first five lines
df.info()                                 # column details

# One important feature is the conversion from string to numeric datatypes!
# For _input_ features, numpy and scikit-learn need numeric datatypes
# You can define a transformation function, to help out...
def transform(s):
    """ from string to number
          setosa -> 0
          versicolor -> 1
          virginica -> 2
    """
    d = { 'unknown':-1, 'setosa':0, 'versicolor':1, 'virginica':2 }
    return d[s]

#
# this applies the function transform to a whole column
#
# df['irisname'] = df['irisname'].map(transform)  # apply the function to the column

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

print("     +++++ Decision Trees +++++\n\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_all = df[ 'irisname' ].values      # individually addressable columns (by name)

X_labeled = X_all[9:,:]  # make the 10 into 0 to keep all of the data
y_labeled = y_all[9:]    # same for this line

#
# we can scramble the data - but only the labeled data!
#
indices = np.random.permutation(len(X_labeled))  # this scrambles the data each time
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

X_train = X_data_full
y_train = y_data_full

#
# some labels to make the graphical trees more readable...
#
print("Some labels for the graphical tree:")
feature_names = ['sepallen', 'sepalwid', 'petallen', 'petalwid']
target_names = ['setosa', 'versicolor', 'virginica']

#
# show the creation of three tree files (at three max_depths)
#
for max_depth in [1,2,3]:
    # the DT classifier
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)

    # train it (build the tree)
    dtree = dtree.fit(X_train, y_train)

    # write out the dtree to tree.dot (or another filename of your choosing...)
    filename = 'tree' + str(max_depth) + '.dot'
    tree.export_graphviz(dtree, out_file=filename,   # the filename constructed above...!
                            feature_names=feature_names,  filled=True,
                            rotate=False, # LR vs UD
                            class_names=target_names,
                            leaves_parallel=True )  # lots of options!
    #
    # Visualize the resulting graphs (the trees) at www.webgraphviz.com
    #
    print("Wrote the file", filename)
    #


#
# cross-validation and scoring to determine parameter: max_depth
#
for max_depth in range(1,12):
    # create our classifier
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
    #
    # cross-validate to tune our model (this week, all-at-once)
    #
    scores = cross_val_score(dtree, X_train, y_train, cv=5)
    average_cv_score = scores.mean()
    print("For depth=", max_depth, "average CV score = ", average_cv_score)
    # print("      Scores:", scores)

# import sys
# print("bye!")
# sys.exit(0)

MAX_DEPTH = 3   # choose a MAX_DEPTH based on cross-validation...
print("\nChoosing MAX_DEPTH =", MAX_DEPTH, "\n")

#
# now, train the model with ALL of the training data...  and predict the unknown labels
#

X_unknown = X_all[0:9,0:4]              # the final testing data
X_train = X_all[9:,0:4]              # the training data

y_unknown = y_all[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_all[9:]                  # the training outputs/labels (known)

# our decision-tree classifier...
dtree = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
dtree = dtree.fit(X_train, y_train)

#
# and... Predict the unknown data labels
#
print("Decision-tree predictions:\n")
predicted_labels = dtree.predict(X_unknown)
answer_labels = answers

#
# formatted printing! (docs.python.org/3/library/string.html#formatstrings)
#
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

#
# feature importances!
#
print()
print("dtree.feature_importances_ are\n      ", dtree.feature_importances_)
print("Order:", feature_names[0:4])


#
# now, show off Random Forests!
#

print("\n\n")
print("     +++++ Random Forests +++++\n\n")

#
# The data is already in good shape -- let's start from the original dataframe:
#
x_all = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_all = df[ 'irisname' ].values      # individually addressable columns (by name)

x_labeled = x_all[9:,:]  # just the input features, X, that HAVE output labels
y_labeled = y_all[9:]    # here are the output labels, y, for X_labeled

#
# we can scramble the data - but only the labeled data!
#
indices = np.random.permutation(len(x_labeled))  # this scrambles the data each time
x_data_full = x_labeled[indices]
y_data_full = y_labeled[indices]

x_train = x_data_full
y_train = y_data_full


#
# cross-validation to determine the Random Forest's parameters (max_depth and n_estimators)
#

#
# Lab task!  Your goal:
#   + loop over a number of values of max_depth (m)
#   + loop over different numbers of trees/n_estimators (n)
#   -> to find a pair of values that results in a good average CV score
#
# use the decision-tree code above as a template for this...
#

# here is a _single_ example call to build a RF:
# m = range(11)
# m = m[1:]
# n = range(102)
# n = n[1::4]
# total_scores = []
# for m_depth in m:
#     average_scores = []
#     for num in n:
#         rforest = ensemble.RandomForestClassifier(max_depth=m_depth, n_estimators=num)
#
#         # an example call to run 5x cross-validation on the labeled data
#         scores = cross_val_score(rforest, X_train, y_train, cv=15)
#         score = scores.mean()
#         average_scores.append(score)
#         # print("CV scores:", scores)
#         # print("CV scores' average:", score)
#         # print("Average Score for # trees", num, "at max depth", m_depth, ":\n", score)
#     total_scores.append(average_scores)
# final_score = []
# for el in total_scores:
#     max_d = total_scores.index(el)
#     local_max = max(el)
#     max_num = el.index(local_max)
#     if not final_score:
#         final_score = [max_d+1, (max_num*4)+1, local_max]
#     elif final_score:
#         if local_max > final_score[2]:
#             final_score = [max_d+1, (max_num*4)+1, local_max]
# print("\nResults:\n", final_score)


# trying max_depth=8 and n_estimators=65 from one run of above code. Can adjust later
# to take the values straight from the above code

#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

x_test = x_all[0:9,0:4]              # the final testing data
x_train = x_all[9:,0:4]              # the training data

y_test = y_all[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_all[9:]                  # the training outputs/labels (known)

# these next lines is where the full training data is used for the model
MAX_DEPTH = 8
NUM_TREES = 65
print()
print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
rforest = rforest.fit(x_train, y_train)

counter = 0
for dt in rforest.estimators_:
    i = rforest.estimators_.index(dt)
    counter += 1
    filename = 'r_tree' + '-' + str(counter) + '_' + str(MAX_DEPTH) + '-' + str(NUM_TREES) + '.dot'
    tree.export_graphviz(rforest.estimators_[i], out_file=filename,   # the filename constructed above...!
                            feature_names=feature_names,  filled=True,
                            rotate=False, # LR vs UD
                            class_names=target_names,
                            leaves_parallel=True )  # lots of options!

# here are some examples, printed out:
print("Random-forest predictions:\n")
predicted_labels = rforest.predict(x_test)
answer_labels = answers  # note that we're "cheating" here!

#
# formatted printing again (see above for reference link)
#
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

#
# feature importances
#
print("\nrforest.feature_importances_ are\n      ", rforest.feature_importances_)
print("Order:", feature_names[0:4])

# The individual trees are in  rforest.estimators_  [a list of decision trees!]
