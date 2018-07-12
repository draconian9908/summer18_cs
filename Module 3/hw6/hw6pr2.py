# coding: utf-8


#
# hw6 problem 2
#

## Problem 2: Analogies!
#

# Run these with
#
# m = read_word2vec_model()
#
# model = read_word2vec_model()
#
def read_word2vec_model():
    """ a function that reads a word2vec model from the file
        "word2vec_model.txt" and returns a model object that
        we will usually name m or model...
    """
    file_name = "word2vec_model.txt"
    from gensim.models import KeyedVectors
    m = KeyedVectors.load_word2vec_format(file_name, binary=False)
    # print("The model built is", m, "\n")
    ## The above line should print
    ## Word2Vec(vocab=43981, size=300, alpha=0.025)
    ## which tells us that the model represents 43981 words with 300-dimensional vectors
    ## The "alpha" is a model-building parameter called the "learning rate."
    ##   Once the model is built, it can't be changed without rebuilding it; we'll leave it.
    return m

m = read_word2vec_model()
# A helper function - are all words in the model?
#
def all_words_in_model( wordlist, model ):
    """ returns True if all w in wordlist are in model
        and False otherwise
    """
    for w in wordlist:
        if w not in model:
            return False
    return True


# Here's a demonstration of the fundamental capability of word2vec on which
#   you'll be building:  most_similar
#
def test_most_similar(model):
    """ example of most_similar """
    print("Testing most_similar on the king - man + woman example...")
    LoM = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
    # note that topn will be 100 below in check_analogy...
    return LoM

# m = read_word2vec_model()
# LoM = test_most_similar(m)
# print(LoM)

#
#
# Start of functions to write + test...
#
#


#
# Write your generate_analogy function
#
#   you will want to base this on the example call made in test_most_similar, above
#
#
print("\n+++ Generate Analogy +++\n")

def generate_analogy(word1, word2, word3, model):
    """ generate_analogy's docstring - be sure to include it!
    """
    print("Testing:\n", word1, "-", word2, "+", word3, ". . .")
    test = all_words_in_model([word1,word2,word3],model)
    if not test:
        print("Error:\nSome word(s) given are not in the model. Analogy cannot be completed.")
    else:
        LoM = model.most_similar(positive=[word3,word1], negative=[word2], topn=100)
        match_word = LoM[0]
        print("=", match_word[0], "\n")

# generate_analogy("cat", "kitten", "puppy", m) # dog
# generate_analogy("bird", "wings", "hands", m) # wildlife
# generate_analogy("person", "man", "woman", m) # someone
# generate_analogy("house", "shelter", "food", m) # kitchen
# generate_analogy("corruption", "money", "chaos", m) # anarchy
# generate_analogy("chaos", "anarchy", "love", m) # loved

print("+++ End Generation +++\n")
#
# Write your check_analogy function
#
print("\n+++ Checking Analogy +++\n")

def check_analogy(word1, word2, word3, word4, model):
    """ check_analogy's docstring - be sure to include it!
    """
    print("Testing:\n", word1, "-", word2, "+", word3, "=", word4)
    test = all_words_in_model([word1,word2,word3,word4],model)
    if not test:
        print("Error:\nSome word(s) given are not in the model. Analogy cannot be completed.")
    else:
        LoM = model.most_similar(positive=[word3,word1], negative=[word2], topn=100)
        LoW = [el[0] for el in LoM]
        if word4 in LoW:
            i = LoW.index(word4)
            pos = 100 - i
        else:
            pos = 0
    print("Score:\n", pos, "\n")

check_analogy("corruption", "money", "chaos", "anarchy", m)
check_analogy("bird", "wings", "hands", "person", m)
check_analogy("king", "man", "woman", "monarch", m)
check_analogy("cat", "kitten", "puppy", "wolf", m)

print("+++ End Check +++\n")
#
# Results and commentary...
#

#
# (1) Write generate_analogy and try it out on several examples of your own
#     choosing (be sure that all of the words are in the model --
#     use the all_words_in_model function to help here)
#
# (2) Report two analogies that you create (other than the ones we looked at in class)
#     that _do_ work reaonably well and report on two that _don't_ work well
#     Finding ones that _do_ work well is more difficult! Maybe in 2025, it'll be the opposite (?)





#
#
# (3) Write check_analogy that should return a "score" on how well word2vec_model
#     does at solving the analogy given (for word4)
#     + it should determine where word4 appears in the top 100 (use topn=100) most-similar words
#     + if it _doens't_ appear in the top-100, it should give a score of 0
#     + if it _does_ appear, it should give a score between 1 and 100: the distance from the
#       _far_ end of the list. Thus, a score of 100 means a perfect score. A score of 1 means that
#       word4 was the 100th in the list (index 99)
#     + Try it out:   check_analogy( "man", "king", "woman", "queen", m ) -> 100
#                     check_analogy( "woman", "man", "bicycle", "fish", m ) -> 0
#                     check_analogy( "woman", "man", "bicycle", "pedestrian", m ) -> 96





#
#
# (4) Create at least five analogies that perform at varying levels of "goodness" based on the
#     check_analogy scoring criterion -- share those (and any additional analysis) with us here!
#
#
