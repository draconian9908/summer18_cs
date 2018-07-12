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
df = pd.read_csv('digits5.csv', header=0)    # read the file w/header row #0
df.head()                                 # first five lines
df.info()                                 # column details
print("\n+++ End of pandas +++\n")


print("+++ Start of numpy/scikit-learn +++\n")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_all = df[ '64' ].values      # individually addressable columns (by name)


X_data_full = X_all[0:,:]  #
y_data_full = y_all[0:]    #


#
# now, model from iris5.py to try DTs and RFs on the digits dataset!
#
