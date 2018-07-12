#
# hw0pr2.py ~ phonebook analysis
#
# Name(s): Lydia
#

#
# be sure your file runs from this location,
# relative to the "phonebook" directories
#

import os
import os.path
import shutil
from hw0pr1a import clean_name



def how_many_txt_files(path):
    """ walks a whole directory structure
        and returns how many txt files are in it!

        call it with: how_many_txt_files(".")

        the (v1) (v2) etc. are different versions, to illustrate
        the process of _trying things out_ and _taking small steps_
    """
    # return 42  # just to check that it's working (v1)

    AllFiles = list(os.walk(path))
    # print(AllFiles)    # just to check out what's up (v2)

    print("AllFiles has length: ", len(AllFiles), "\n")

    total = 0

    for item in AllFiles:
        # print("item is", item, "\n")   # (v3)
        foldername, LoDirs, LoFiles = item   # cool!
        print("In", foldername, "there are", end=" ")

        count = 0
        for filename in LoFiles:
            if filename[-3:] == "txt":
                count += 1
        total += count
        print(count, ".txt files")

    return total


def max_dir_depth(path):
    """ walks a whole directory structure
        and returns:
        the maximum depth of directories for the entire folder;
        the path to the deepest directory.

        call with: max_dir_depth(".")
    """
    AllFiles = list(os.walk(path))
    depths = []
    dirs = []

    for item in AllFiles:
        foldername, LoDirs, LoFiles = item
        depths.append(foldername.count("/"))
        dirs.append(foldername)
    maximum = max(depths)
    i = depths.index(maximum)
    return [depths[i],dirs[i]]


def ten_digit_nums(path):
    """ walks a whole directory structure
        and returns:
        the number of phone numbers with 10 digits;
        the number of 10 digit phone numbers with 909 area code.

        call with: ten_digit_nums(".")
    """
    AllFiles = list(os.walk(path))
    num_count = 0
    area_count = 0
    numbers = []

    for item in AllFiles:
        foldername, LoDirs, LoFiles = item
        for filename in LoFiles:
            fullname = foldername + "/" + filename
            f = open(fullname, "r", encoding="latin1")
            contents = f.read()
            num_digits = 0
            temp = []
            for element in contents:
                if element.isdigit():
                    num_digits += 1
                    temp.append(element)
            if num_digits == 10:
                num_count += 1
                numbers.append(temp)
    for num in numbers:
        corrected_num = "".join(num)
        # print(corrected_num[:3])
        if corrected_num[:3] == "909":
            area_count += 1
    return [num_count, area_count]

def have_last_name(path,name):
    """ walks a whole directory structure
        and returns the number of people with the last name `name`

        call with: have_last_name(".","Davis")
    """
    AllFiles = list(os.walk(path))
    count = 0

    for item in AllFiles:
        foldername, LoDirs, LoFiles = item
        for filename in LoFiles:
            fullname = foldername + "/" + filename
            f = open(fullname, "r", encoding="latin1")
            contents = f.read()
            cleaned = clean_name(contents)
            edited = cleaned.lstrip(" ")
            if edited.find(",") > 0:
                i = edited.find(",")
                lastname = edited[:i]
                # print(lastname)
                if lastname == name:
                    count += 1
            else:
                i = edited.find(" ")
                lastname = edited[i+1:]
                # print(lastname)
                if lastname == name:
                    count += 1
    return count



# __main__ function
def main():
    """ overall function to run all examples """

    print("Start of main()\n")

    # determine total number of txt files
    num_txt_files = how_many_txt_files(".")
    print("num_txt_files in . is", num_txt_files)

    # determine maximum directory depth and path to deepest directory
    max_depth_dir = max_dir_depth(".")
    print("max_depth in . is", max_depth_dir[0], "and the deepest path is", max_depth_dir[1])

    # determine total number of 10 digit phone numbers and number of those in 909 area
    num_ten_digit_numbers = ten_digit_nums(".")
    print("num_ten_digit_numbers in . is", num_ten_digit_numbers[0], "and number in of those in 909 area is", num_ten_digit_numbers[1])

    # determine total number of people with last name Davis
    num_lastname_Davis = have_last_name(".","Davis")
    print("num_lastname_Davis in . is", num_lastname_Davis)

    # determine total number of people with last name Scarboro
    num_lastname_Salen = have_last_name(".","Salen")
    print("num_lastname_Salen in . is", num_lastname_Salen)

    print("End of main()\n")


if __name__ == "__main__":
    main()
