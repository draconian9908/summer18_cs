#
#
# titanic.py
#
#

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd

print("+++ start of pandas +++\n")
# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic.csv', header=0)
df.head()
df.info()
for_write = df.copy()
# let's drop columns with too few values or that won't be meaningful
# Here's an example of dropping the 'body' column:
df = df.drop('body', axis=1)  # axis = 1 means column
df = df.drop('cabin', axis=1)
df = df.drop('boat', axis=1)
df = df.drop('home.dest', axis=1)
df = df.drop('ticket', axis=1)
df = df.drop('name', axis=1)

for_write = for_write.drop('body', axis=1)  # axis = 1 means column
for_write = for_write.drop('cabin', axis=1)
for_write = for_write.drop('boat', axis=1)
for_write = for_write.drop('home.dest', axis=1)
for_write = for_write.drop('ticket', axis=1)
for_write = for_write.dropna()
# let's drop all of the rows with missing data:
df = df.dropna()
print("Updated dataframe:\n")
# let's see our dataframe again...
# I ended up with 1001 rows (anything over 500-600 seems reasonable)
df.head()
df.info()
print("\n\n")

to_write = for_write.iloc[:,1:5]
# You'll need conversion to numeric datatypes for all input columns
#   Here's one example
#
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':1, 'female':2 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column

def tr_em(s):
    """ string to number
    """
    d = {'C':0, 'Q':1, 'S':2}
    return d[s]

df['embarked'] = df['embarked'].map(tr_em)


print("Converted dataframe:\n")
# let's see our dataframe again...
df.head()
df.info()
print("\n")


# you will need others!


print("+++ end of pandas +++\n\n")

# import sys
# sys.exit(0)

print("+++ start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array

# extract the underlying data with the values attribute:
x_data_full = df.drop('survived', axis=1).values        # everything except the 'survival' column
y_data_full = df[ 'survived' ].values      # also addressable by column name(s)

x_data_full[0] *= 5
x_data_full[1] *= 5
x_data_full[2] *= 100
x_data_full[5] *= 0.125
#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
x_data = x_data_full[42:]
y_data = y_data_full[42:]


# feature engineering...
#X_data[:,0] *= 100   # maybe the first column is worth much more!
#X_data[:,3] *= 100   # maybe the fourth column is worth much more!




#
# the rest of this model-building, cross-validation, and prediction will come here:
#     build from the experience and code in the other two examples...
#

from sklearn.neighbors import KNeighborsClassifier

def knn_sim(num_neighbors,runs=50):
    max_runs = runs
    train_array = np.zeros(len(num_neighbors))
    test_array = np.zeros(len(num_neighbors))
    while runs > 0:
        train_average = []
        test_average = []
        for num in num_neighbors:
            i = num_neighbors.index(num)
            counter = 0
            train_results = []
            test_results = []
            while counter <= 9:
                knn = KNeighborsClassifier(n_neighbors=num)   # 7 is the "k" in kNN

                #
                # cross-validate (use part of the training data for training - and part for testing)
                #   first, create cross-validation data (here 3/4 train and 1/4 test)
                cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
                    cross_validation.train_test_split(x_data, y_data, test_size=0.25) # random_state=0

                # fit the model using the cross-validation data
                #   typically cross-validation is used to get a sense of how well it works
                #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
                knn.fit(cv_data_train, cv_target_train)
                # print("KNN cv training-data score:", knn.score(cv_data_train,cv_target_train))
                # print("KNN cv testing-data score:", knn.score(cv_data_test,cv_target_test), "\n")
                train_results.append(knn.score(cv_data_train,cv_target_train))
                test_results.append(knn.score(cv_data_test,cv_target_test))
                counter += 1

            train_average.append(sum(train_results) / 10)
            test_average.append(sum(test_results) / 10)
        train_array = np.add(train_array,np.array(train_average))
        test_array = np.add(test_array,np.array(test_average))
        runs -= 1
    i_train = np.argmax(train_array / max_runs)
    i_test = np.argmax(test_array / max_runs)
    max_k = num_neighbors[i_test]
    print("Result:\n", max_k)

# knn_sim([1,3,5,7,9,11,15,21,33,45,67,91,303,677])
# best k: 15!

knn = KNeighborsClassifier(n_neighbors=15)   # 7 is the "k" in kNN

cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
    cross_validation.train_test_split(x_data, y_data, test_size=0.25) # random_state=0

knn.fit(cv_data_train, cv_target_train)

test_results = knn.score(cv_data_test,cv_target_test)
print("Accuracy:\n", test_results, "\n")

print("+++ end of numpy/scikit-learn +++\n")

knn = KNeighborsClassifier(n_neighbors=15)
# this next line is where the full training data is used for the model
knn.fit(x_data, y_data)
print("\nCreated and trained a knn classifier")  #, knn
results = knn.predict(x_data_full[0:42])
y_data_full[:42] = results
to_write['survived'] = pd.DataFrame(y_data_full)
to_write.to_csv('completed_titanic.csv')

# print('Results:\n')#, results)
# for elem in results:
#     if elem == 0:
#         print("Killed")
#     elif elem == 1:
#         print("Survived")
#     else:
#         print("f")


"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?
  + how high were you able to get the average cross-validation (testing) score?



Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:




And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

Past those labels (just labels) here:
You'll have 10 lines:



"""
