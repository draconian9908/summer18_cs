#
#
# digits.py
#
#

import numpy as np
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('digits.csv', header=0)
df.head()
df.info()

# Convert feature columns as needed...
# You may to define a function, to help out:
def transform(s):
    """ from number to string
    """
    return 'digit ' + str(s)
    
df['label'] = df['64'].map(transform)  # apply the function to the column
print("+++ End of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ Start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array
X_data = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_data = df[ 'label' ].values      # also addressable by column name(s)

#
# you can divide up your dataset as you see fit here...
#


#
# feature display - use %matplotlib to make this work smoothly
#
from matplotlib import pyplot as plt

def show_digit( Pixels ):
    """ input Pixels should be an np.array of 64 integers (from 0 to 15) 
        there's no return value, but this should show an image of that 
        digit in an 8x8 pixel square
    """
    print(Pixels.shape)
    Patch = Pixels.reshape((8,8))
    plt.figure(1, figsize=(4,4))
    plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # cm.gray_r   # cm.hot
    plt.show()
    
# try it!
row = 3
Pixels = X_data[row:row+1,:]
show_digit(Pixels)
print("That image has the label:", y_data[row])











#
# feature engineering...
#





#
# here, you'll implement the kNN model
#


#
# run cross-validation
#


#
# and then see how it does on the two sets of unknown-label data... (!)
#





"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?
  + how smoothly were you able to adapt from the iris dataset to here?
  + how high were you able to get the average cross-validation (testing) score?



Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:




And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

Past those labels (just labels) here:
You'll have 10 lines:



"""