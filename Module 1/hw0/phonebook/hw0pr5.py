#
# hw0pr5.py ~ writing files
#
# Name: Lydia
#
#

import os
from hw0pr1a import clean_digits, clean_name


def clean_lastname(contents):
    """ takes a string and returns only the last name """
    edited = clean_name(contents)
    names = edited.lstrip(" ")
    if names.find(",") > 0:
        i = names.find(",")
        lastname = names[:i]
    else:
        i = names.find(" ")
        lastname = names[i+1:]

    return lastname


def clean_firstname(contents):
    """ takes a string and returns only the first name """
    edited = clean_name(contents)
    names = edited.lstrip(" ")
    if names.find(",") > 0:
        i = names.find(",")
        firstname = names[i+2:]
    else:
        i = names.find(" ")
        firstname = names[:i]

    return firstname



def write_file(path):
    """ opens/creates a csv file,
        uses other functions to parse phonebook data into separate pieces,
        writes those pieces to csv file.

        csv file: phone.csv
        data: first name, last name, phone number
    """
    file = open(path + "/phone.csv", mode="w", encoding="latin1")
    AllFiles = list(os.walk(path))

    for item in AllFiles:
        foldername, LoDirs, LoFiles = item
        for filename in LoFiles:
            if filename[-2:] != "py":
                fullname = foldername + "/" + filename
                f = open(fullname, "r", encoding="latin1")
                contents = f.read()
                number = clean_digits(contents)
                lastname = clean_lastname(contents)
                firstname = clean_firstname(contents)
                text = lastname + ", " + firstname + ", " + number + "\n"
                print(text)
                file.write(text)

    file.close()



def main():
    """ organize and run all functions """
    print("Starting Main\n")

    write_file(".")

    print("\nEnding Main")


if __name__ == "__main__":
    main()
