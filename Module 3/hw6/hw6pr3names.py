# coding: utf-8

# Problem 4:  machine-learning and NLP
#
# Name-classification
#
# This problem asks you to model whether a name is more likely to be
# identified as female or male, based on user-defined features...
#
# It creates a confusion matrix of the results -- and shares some of the mis-classifications
#

"""
###########
#
# Note: for the example and problem below, you will need
#       three of the "language corpora" (large sourcetexts)
#       from NLTK. To make sure you've downloaded them,
#       run the following:
#
# import nltk
# nltk.download('names')
# nltk.download('movie_reviews')
# nltk.download('opinion_lexicon')
#
# others are available, e.g.
# nltk.download('stopwords')
#
###########
"""

## Import all of the libraries and data that we will need.
import nltk
from nltk.corpus import names  # see the note on installing corpora, above
from nltk.corpus import opinion_lexicon
from nltk.corpus import movie_reviews

import random
import math

from sklearn.feature_extraction import DictVectorizer
import sklearn
import sklearn.tree
from sklearn.metrics import confusion_matrix
from collections import defaultdict


#
# experiments showing how the feature vectorizer in scikit-learn works...
#
TRY_FEATURE_VECTORIZER = False
if TRY_FEATURE_VECTORIZER == True:
    # converts from dictionaries to feature arrays (vectors)
    v = DictVectorizer(sparse=False)
    FEATS = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    X = v.fit_transform(FEATS)

    print("FEATS are\n", FEATS, "\n")
    print("v is", v)
    print("X is\n", X, "\n")
    print("Forward: ", v.transform({'foo': 4, 'unseen_feature': 3}), "\n")
    print("Inverse: ", v.inverse_transform(X),"\n")



#
# Language-modeling example:  is a name female or male?
#

# a boolean to turn on/off the name-classifier portion of the code...
RUN_NAME_CLASSIFIER = True
if RUN_NAME_CLASSIFIER == True:

    ## Read all of the names in from the nltk corpus and organize them with labels
    female_names = [(name,'f') for name in names.words('female.txt')]
    male_names = [(name,'m') for name in names.words('male.txt')]
    labeled_names = male_names + female_names

    ## Shuffle all of the labeled names
    # random.seed(0)  # RNG seed: use this for "reproduceable random numbers"
    random.shuffle(labeled_names)


    #
    # Here is where the "art" of feature-development happens...
    #

    #
    ## Define the feature function; we'll modify this to improve
    ## the classifier's performance.
    #
    def gender_features(word):
        """ feature function for the female/male names example
            This function should return a dictionary of features,
            which can really be anything we might compute from the input word...
        """
        features = {}
        features['last_letter'] = word[-1].lower()
        return features



    #
    # This is a better set of features --  Try it!
    #
    def gender_features_2(word):
        """ feature function for the female/male names example
            This function should return a dictionary of features,
            which can really be anything we might compute from the input word...
        """
        features = {}
        features['last_letter'] = word[-1].lower()
        features['first_letter'] = word[0].lower()
        return features


    #
    # This is an EVEN better set of features --  Try it!
    #
    def gender_features_3(word):
        """ feature function for the female/male names example
            This function should return a dictionary of features,
            which can really be anything we might compute from the input word...
        """
        features = defaultdict(int)

        features['s'] = word.count('s')
        features['y'] = word.count('y')
        features['last-letter'] = word[-1]

        for l in 'aeiouy':
            features['vowels'] += word.count(l)

        for l0 in 'aeiouy':
            for l1 in 'aeiouy':
                if (l0+l1 in word):
                    features['double-vowel'] += 1

        return features


    def gender_features_mine(word):
        """ feature function for the female/male names example
            This function should return a dictionary of features,
            which can really be anything we might compute from the input word...
        """
        features = defaultdict(int)

        features['s'] = word.count('s')
        features['y'] = word.count('y')
        features['last-letter'] = word[-1]

        for l in 'aeiouy':
            features['vowels'] += word.count(l)

        for l0 in 'aeiouy':
            for l1 in 'aeiouy':
                if (l0+l1 in word):
                    features['double-vowel'] += 1

        for l2 in 'bcdfghjklmnpqrstvwxz':
            for l3 in 'bcdfghjklmnpqrstvwxyz':
                if (l2+l3 in word):
                    features['double-consonant'] += 1

        features['l'] = word.count('l')
        features['i'] = word.count('i')
        features['f'] = word.count('f')

        if 'f' in word[1:]:
            features['internal-f'] = 1

        for l4 in 'a':
            if l4 == word[-1]:
                features['fem-end'] = 1

        for l5 in 'o':
            if l5 == word[-1]:
                features['masc-end'] = 1

        for l6 in 'blmn':
            features['round'] += word.count(l6)
        for l7 in 'kpt':
            features['sharp'] += word.count(l7)

        return features


    #
    # Here is where the _reproduceable_ part of the classification happens:
    #

    ## Compute features and extract labels and names for the whole dataset
    features = [gender_features_mine(name) for (name, gender) in labeled_names]
    labels = [gender for (name, gender) in labeled_names]
    names = [name for (name, gender) in labeled_names]


    ## Change the dictionary of features into an array with DictVectorizer
    ## then, create our vector of input features, X  (our usual name for it...)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(features)


    ## Split the input features into train, devtest, and test sets
    X_test = X[:500,:]
    X_devtest = X[500:1000,:]  # sometimes called a "validation" set
    X_train = X[1000:,:]                                                    # originally 1500

    ## Split the output (labels) into train, devtest, and test sets
    y_test = labels[:500]
    y_devtest = labels[500:1000]
    y_train = labels[1000:]

    ## Split the names themselves into train, devtest, and test sets (for reference)
    names_test = names[:500]
    names_devtest = names[500:1000]
    names_train = names[1000:]



    ############
    #
    # All of the set-up for the name-classification task is now ready...
    #
    ############

    ## Train a decision tree classifier
    #
    dt = sklearn.tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)  # fit with the training data!

    ## Evaluate on the devtest set (with known labels) and report the accuracy
    print("Score on devtest set: ", dt.score(X_devtest, y_devtest))

    ## Predict the results for the devtest set and show the confusion matrix.
    y_guess = dt.predict(X_devtest)
    CM = confusion_matrix(y_guess, y_devtest,  labels=['f','m'])
    print("Confusion Matrix:\n", CM)

    ## a function to predict individual names...
    def classify( name, model, feature_vectorizer, feature_function ):
        """ predictor! """
        features = feature_function(name)
        X = feature_vectorizer.transform(features)
        guess = model.predict(X)[0]
        return guess

    # Example to try out...
    LoN = [ "Zach", "Julie", "Colleen", "Melissa", "Ran", "Geoff", "Bob", "Jessica", "Lydia", "Everardo", "Evelyne" ]
    for name in LoN:
        guess = classify( name, dt, v, gender_features )
        print(guess,name)


    ## Get a list of errors to examine more closely.
    errors = []
    for i in range(len(names_devtest)):
        this_name = names_devtest[i]
        this_features = X_devtest[i:i+1,:]   # slice of all features for name i
        this_label = y_devtest[i]
        guess = dt.predict(this_features)[0] # predict (guess) from the features...
        #
        # if the guess was incorrect (wrong label), remember it!
        #
        if guess != this_label:
            errors.append((this_label, guess, this_name))


    # Now, print out the results: the incorrect guesses
    # Create a flag to turn this printing on/off...
    #
    PRINT_ERRORS = False
    if PRINT_ERRORS == True:
        SE = sorted(errors)
        print("There were", len(SE), "errors:")
        print('Name: guess (actual)')
        num_to_print = 10
        for (actual, guess, name) in SE:
            if actual == 'm' and guess == 'f': # adjust these as needed...
                print('  {0:>10}: {1} ({2})'.format(name, guess, actual))
                num_to_print -= 1
                if num_to_print == 0: break
        print()


    ## Finally, score on the test set:
    print("Score on test set: ", dt.score(X_test, y_test))

    ## Don't actually tune the algorithm for the test set, however!

    ## Reflection / Analysis for the name-classification example
    #

    """
    Try different features, e.g.,
    + other letters
    + vowel/nonvowel in certain positions
    + presence or absence of one (or more)-letter strings ...
    """
