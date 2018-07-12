#
# titanic5: modeling the Titanic data with DTs and RFs
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
# The "answers" to the 30 unlabeled passengers:
#
answers = [0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,
            1,0,1,1,1,1,0,0,0,1,1,0,1,0]

#

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('titanic5.csv', header=0)    # read the file w/header row #0
#
# drop columns here
#
df = df.drop('name', axis=1)  # axis = 1 means column
df = df.drop('cabin', axis=1)
df = df.drop('home.dest', axis=1)
df = df.drop('ticket', axis=1)

df = df.dropna()

df.head()                                 # first five lines
df.info()                                 # column details

# One important one is the conversion from string to numeric datatypes!
# You need to define a function, to help out...
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column

def tr_em(s):
    """ string to number
    """
    d = {'C':0, 'Q':1, 'S':2}
    return d[s]

df['embarked'] = df['embarked'].map(tr_em)

#
# end of conversion to numeric data...
#
print("\n+++ End of pandas +++\n")

#

print("+++ Start of numpy/scikit-learn +++\n")

print("     +++++ Decision Trees +++++\n\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
x_all = df.drop('survived', axis=1).values
y_all = df[ 'survived' ].values

x_labeled = x_all[30:]
y_labeled = y_all[30:]

indices = np.random.permutation(len(x_labeled))  # this scrambles the data each time
x_data_full = x_labeled[indices]
y_data_full = y_labeled[indices]

x_train = x_data_full
y_train = y_data_full

print("Some labels for the graphical tree:")
feature_names = ['pclass','sex','age','sibsp','parch','fare','embarked']
target_names = [0,1]
for_file = ['0','1']
print("Features:\n", feature_names, "\nTargets:\n", target_names)

""" DT Model """

# for max_depth in [3,4,5,6,7,10,100]:
#     # the DT classifier
#     dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
#
#     # train it (build the tree)
#     dtree = dtree.fit(x_train, y_train)
#
#     # write out the dtree to tree.dot (or another filename of your choosing...)
#     filename = 'tree' + str(max_depth) + '.dot'
#     tree.export_graphviz(dtree, out_file=filename,   # the filename constructed above...!
#                             feature_names=feature_names,  filled=True,
#                             rotate=False, # LR vs UD
#                             class_names=for_file,
#                             leaves_parallel=True )  # lots of options!
#     print("Wrote the file", filename)

all_scores = []
for max_depth in range(1,12):
    # create our classifier
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
    #
    # cross-validate to tune our model (this week, all-at-once)
    #
    scores = cross_val_score(dtree, x_train, y_train, cv=5)
    average_cv_score = scores.mean()
    all_scores.append(average_cv_score)
    print("For depth=", max_depth, "average CV score = ", average_cv_score)

MAX_DEPTH = max(all_scores)

x_unknown = x_all[0:30]
x_train = x_all[30:]

y_unknown = y_all[0:30]
y_train = y_all[30:]

dtree = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
dtree = dtree.fit(x_train, y_train)

print("\nDecision-tree predictions:\n")
predicted_labels = dtree.predict(x_unknown)
answer_labels = answers

# formatted printing! (docs.python.org/3/library/string.html#formatstrings)
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

# feature importances!
print()
print("dtree.feature_importances_ are\n      ", dtree.feature_importances_)
print("Order:", feature_names)

""" RF Model """

print("\n\n")
print("     +++++ Random Forests +++++\n\n")

x_all = df.drop('survived', axis=1).values
y_all = df[ 'survived' ].values

x_labeled = x_all[30:]
y_labeled = y_all[30:]

indices = np.random.permutation(len(x_labeled))  # this scrambles the data each time
x_data_full = x_labeled[indices]
y_data_full = y_labeled[indices]

x_train = x_data_full
y_train = y_data_full

m = range(21)
m = m[1:]
n = range(162)
n = n[1::4]
total_scores = []
for m_depth in m:
    average_scores = []
    for num in n:
        rforest = ensemble.RandomForestClassifier(max_depth=m_depth, n_estimators=num)

        # an example call to run 5x cross-validation on the labeled data
        scores = cross_val_score(rforest, x_train, y_train, cv=7)
        score = scores.mean()
        average_scores.append(score)
        # print("CV scores:", scores)
        # print("CV scores' average:", score)
        # print("Average Score for # trees", num, "at max depth", m_depth, ":\n", score)
    total_scores.append(average_scores)
final_score = []
for el in total_scores:
    max_d = total_scores.index(el)
    local_max = max(el)
    max_num = el.index(local_max)
    if not final_score:
        final_score = [max_d+1, (max_num*4)+1, local_max]
    elif final_score:
        if local_max > final_score[2]:
            final_score = [max_d+1, (max_num*4)+1, local_max]
print("Results:\n", final_score)

x_test = x_all[0:30]
x_train = x_all[30:]

y_test = y_all[0:30]
y_train = y_all[30:]

MAX_DEPTH = final_score[0]
NUM_TREES = final_score[1]
print()
print("Using MAX_DEPTH =", MAX_DEPTH, "and NUM_TREES =", NUM_TREES)
rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
rforest = rforest.fit(x_train, y_train)

# counter = 0
# for dt in rforest.estimators_:
#     i = rforest.estimators_.index(dt)
#     counter += 1
#     filename = 'r_tree' + '-' + str(counter) + '_' + str(MAX_DEPTH) + '-' + str(NUM_TREES) + '.dot'
#     tree.export_graphviz(rforest.estimators_[i], out_file=filename,   # the filename constructed above...!
#                             feature_names=feature_names,  filled=True,
#                             rotate=False, # LR vs UD
#                             class_names=for_file,
#                             leaves_parallel=True )  # lots of options!

# here are some examples, printed out:
print("Random-forest predictions:\n")
predicted_labels = rforest.predict(x_test)
answer_labels = answers  # note that we're "cheating" here!

# formatted printing again (see above for reference link)
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

correct = 0
for p, a in zip( predicted_labels, answer_labels ):
    if p == a:
        correct += 1
accuracy = correct/len(answer_labels) * 100
print("\nFinal Accuracy:\n", accuracy)

# feature importances
print("\nrforest.feature_importances_ are\n", rforest.feature_importances_)
print("Order:\n", feature_names)

#
# now, building from iris5.py and/or digits5.py
#      create DT and RF models on the Titanic dataset!
#      Goal: find feature importances ("explanations")
#      Challenge: can you get over 80% CV accuracy?
#
