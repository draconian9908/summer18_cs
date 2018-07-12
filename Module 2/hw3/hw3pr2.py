#
# hw3pr2.py
#
# Person or machine?  The rps-string challenge...
#
# This file should include your code for
#   + extract_features( rps ),               returning a dictionary of features from an input rps string
#   + score_features( dict_of_features ),    returning a score (or scores) based on that dictionary
#   + read_data( filename="rps.csv" ),       returning the list of datarows in rps.csv
#
# Be sure to include a short description of your algorithm in the triple-quoted string below.
# Also, be sure to include your final scores for each string in the rps.csv file you include,
#   either by writing a new file out or by pasting your results into the existing file
#   And, include your assessment as to whether each string was human-created or machine-created
#
#

"""
Short description of (1) the features you compute for each rps-string and
      (2) how you score those features and how those scores relate to "humanness" or "machineness"





"""


# Here's how to machine-generate an rps string.
# You can create your own human-generated ones!

import random

def gen_rps_string( num_characters ):
    """ return a uniformly random rps string with num_characters characters """
    result = ''
    for i in range( num_characters ):
        result += random.choice( 'rps' )
    return result

# Here are two example machine-generated strings:
rps_machine1 = gen_rps_string(200)
rps_machine2 = gen_rps_string(200)
# print those, if you like, to see what they are...




from collections import defaultdict
import csv
from itertools import groupby

#
# extract_features( rps ):   extracts features from rps into a defaultdict
#
def extract_features( rps ):
    """ <include a docstring here!>
    """
    d = defaultdict( float )  # other features are reasonable
    # number_of_s_es = rps.count('s')  # counts all of the 's's in rps
    # d['s'] = 42                      # doesn't use them, however
    num_rps = []
    num_s = []
    num_r = []
    num_p = []
    for lines in rps:
        print("\nNew Line!")
        line = lines[1]
        count_rps = line.count('rps')  # counts number of times 'rps' appears, in that exact order (h)
        num_rps.append(count_rps)

        # count_s =
        max_r, max_p, max_s = con_count(line)
        num_r.append(max_r)
        num_p.append(max_p)
        num_s.append(max_s)

    d['rps'] = num_rps
    d['r'] = num_r
    d['p'] = num_p
    d['s'] = num_s
    # print("\nCurrent Dictionary:\n", d)
    return d   # return our features... this is unlikely to be very useful, as-is



def con_count(s):
    r_count = []
    p_count = []
    s_count = []
    for k, g in groupby(s):
        if k == 'r':
            r_count.append(sum(1 for _ in g))
        elif k == 'p':
            p_count.append(sum(1 for _ in g))
        elif k == 's':
            s_count.append(sum(1 for _ in g))

    if r_count:
        max_r = max(r_count)
    if not r_count:
        max_r = 0
    if p_count:
        max_p = max(p_count)
    if not p_count:
        max_p = 0
    if s_count:
        max_s = max(s_count)
    if not s_count:
        max_s = 0

    return [max_r, max_p, max_s]




#
# score_features( dict_of_features ): returns a score based on those features
#
def score_features( dict_of_features ):
    """ <include a docstring here!>
    """
    d = dict_of_features
    score_rps = []
    for num in d['rps']:
        if num >= 10:
            score_rps.append(num * 1.1)
        else:
            score_rps.append(num * -1.1)
    for num in d['r']:
        if num > 5:
            score_r.append(num * 2)
        else:
            score_r.append(num * )
    # random_value = random.uniform(0,1)
    # score = d['s'] * random_value
    return score_rps   # return a humanness or machineness score







#
# read_data( filename="rps.csv" ):   gets all of the data from "rps.csv"
#
def read_data( filename="rps18.csv" ):
    """ <include a docstring here!>
    """
    try:
        csvfile = open( filename, newline='' )  # open for reading
        csvrows = csv.reader( csvfile )              # creates a csvrows object

        all_rows = []                               # we need to read the csv file
        for row in csvrows:                         # into our own Python data structure
            all_rows.append( row )                  # adds only the word to our list

        del csvrows                                  # acknowledge csvrows is gone!
        csvfile.close()                              # and close the file
        return all_rows                              # return the list of lists

    except FileNotFoundError as e:
        print("File not found: ", e)
        return []


def human_machine(LoS):
    LoR = []
    counter = 0
    h_count = 0
    m_count = 0
    for score in LoS:
        row = [counter]
        counter += 1
        if score >= 0:
            row.append('human')
            h_count += 1
        else:
            row.append('machine')
            m_count += 1
        LoR.append(row)
    print("Number Human:\n" + str(h_count) + "\nNumber Machine:\n" + str(m_count))
    return LoR


def write_csv(LoR, filename='rps.csv'):
    try:
        csvfile = open(filename, "w", newline='')
        filewriter = csv.writer(csvfile,delimiter=",")
        for row in LoR:
            filewriter.writerow(row)
        csvfile.close()

    except:
        print("File", filename, "could not be opened for writing...")


def main():
    LoL = read_data()
    LoF = extract_features(LoL)
    LoS = score_features(LoF)
    LoR = human_machine(LoS)
    write_csv(LoR)



if __name__ == "__main__":
    main()



#
# you'll use these three functions to score each rps string and then
#    determine if it was human-generated or machine-generated
#    (they're half and half with one mystery string)
#
# Be sure to include your scores and your human/machine decision in the rps.csv file!
#    And include the file in your hw3.zip archive (with the other rows that are already there)
#
