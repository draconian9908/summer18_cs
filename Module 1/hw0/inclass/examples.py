#
# cs35 examples in-class  1/22/18
#

#
# Note! This script does NOT do a great job of explaining
#  what it's doing - this is something your scrips will be
#  able to do far better :-) !
#


"""
DEMO 1:

Building paths to directories and accessing their contents

the os and os.path libraries are documented here:
  https://docs.python.org/3/library/os.html
  https://docs.python.org/3/library/os.path.html
"""

import os
import os.path
import shutil

def directory_examples():
    """ examples for directory navigation and contents... """

    # get current working directory
    original_dir = os.getcwd()
    print("original_dir is", original_dir)

    # construct a path to a directory
    path = os.path.join(original_dir, "addresses")  # the path to the addresses directory
    print("now, I can access", path)

    # get a listing of all of the contents of the directory
    DirContents = os.listdir(path)
    print("DirContents:", DirContents)

    # change back!
    path = os.path.dirname(path) # back to original
    print("and now, path points back to ", path)

    # make a directory called not-hp
    os.mkdir(os.path.join(path, "not-hp"))
    print("Now the contents include not-hp: ", os.listdir(path))

    # move samiam.txt into not-hp
    shutil.move(os.path.join(path, "hp/samiam.txt"), os.path.join(path, "not-hp"))

    # move samiam back to hp
    shutil.move(os.path.join(path, "not-hp/samiam.txt"), os.path.join(path, "hp"))

    # delete the not-hp directory
    os.rmdir(os.path.join(path, "not-hp"))

    # loop through every directory in a tree and print its subdirectories and files
    # os.walk(path) will be useful for the hard drive scavenger hunt!
    for current_directory, sub_directories, files in os.walk(path):
        print()
        print(current_directory)
        for file_name in files:
            print(" -" + file_name)
        for sub_directory_name in sub_directories:
            print(" +" + sub_directory_name)


    # +++ Challenge: access the hp directory and list its contents:
    # +++ Challenge: walk the inclass directory and count the number of files



"""
DEMO 2:

Opening files and reading their contents

Documentation:
  https://docs.python.org/3.3/tutorial/inputoutput.html#reading-and-writing-files
  [Extra] file encodings:  https://docs.python.org/3/library/codecs.html
"""

# Assuming path points to the hp directory, let's open filename ("samiam.txt")
#
def file_examples(path, filename):
    """ examples of file reading and exceptions """
    filepath = os.path.join(path, filename)

    try:
        f = open(filepath,"r", encoding="latin1") # latin1 is a very safe encoding
        data = f.read()   # read all of the file's data
        f.close()         # close the file

    except PermissionError:  # example of "exceptions": atypical errors
        print("file", filename, "couldn't be opened: permission error")
        data = ""

    except UnicodeDecodeError:
        print("file", filename, "couldn't be opened: encoding error")
        data = "" # no data

    except FileNotFoundError:  # try it with and without this block...
        print("file", filename, "couldn't be opened: not found!")
        print("Check that you have the correct path...")
        data = ""

    # We return the data we obtained in trying to open the file
    #print("File data:", data)
    return data    # remember print and return are different!

    # ++ Challenge: loop over all of the files in this directory, add up their contents
    #            and return the results (helpful for problem #2)

    # ++ Challenge: change the function to include an input filename
    #            and return the data from that file (also helpful for #2 and #3)



"""
DEMO 3:

Text analysis of data obtained from a file...

Here we introduce the _much_ nicer alternative to dictionaries, called
    default dictionaries, or defaultdict, with documentation here:
    https://docs.python.org/3/library/collections.html#collections.defaultdict

In addition, we introduce some useful parts of the string library:
    https://docs.python.org/3.1/library/string.html
    Methods such as s.lower(), s.upper(), s.split(), ... are wonderful!
"""

from collections import defaultdict      # be sure to import it!


# This is a function that counts all of the 'A's and 'a's in the input
#
def count_a( data ):
    """ this function returns a default dictionary that contains
        the key 'a': its value is equal to the number of 'a's in the input, data
        NOTE: everything is lower-cased, so this really counts 'A's and 'a's
    """
    counts = defaultdict(int)
    # data = data.lower()       # lower case all of the data
    for c in data:            # loop over each character in the data
        if c == 'a' or c == 'A':
            counts['a'] += 1

    return counts





#
# Here is the main() function to organize our work...
#
def main():
    """ This "main" function will read the file "samiam.txt"
        and then count it's A's (or a's) and print the total number present
    """
    # sign on
    print("\n\nStart of main()\n\n")

    # examples of navigating among files and directories
    # it does its own printing
    directory_examples()

    # example 2
    print("\n\nExample 2: analyzing file contents\n\n")

    # Here's another example - creating a path
    # opening a file, and analyzing its contents...

    # we build a path to the hp directory
    ourpath = os.getcwd()
    path = os.path.join(ourpath, "hp" ) # add "hp"
    filename = "samiam.txt"
    data = file_examples(path, filename)
    counts = count_a( data )
    num_of_a = counts['a']
    print("The number of a's in samiam.txt is", num_of_a)

    # sign off
    print("\n\nEnd of main()\n\n")




# This runs main() when the file is executed.

if __name__ == '__main__':
    main()



