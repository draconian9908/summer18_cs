#
# starter file for hw1pr2, cs35 spring 2017...
#

import csv
from collections import *

#
# readcsv is a starting point - it returns the rows from a standard csv file...
#
def readcsv( csv_file_name ):
    """ readcsv takes as
         + input:  csv_file_name, the name of a csv file
        and returns
         + output: a list of lists, each inner list is one row of the csv
           all data items are strings; empty cells are empty strings
    """
    try:
        csvfile = open( csv_file_name, newline='' )  # open for reading
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



#
# write_to_csv shows how to write that format from a list of rows...
#  + try   write_to_csv( [['a', 1 ], ['b', 2]], "smallfile.csv" )
#
def write_to_csv( list_of_rows, filename ):
    """ readcsv takes as
         + input:  csv_file_name, the name of a csv file
        and returns
         + output: a list of lists, each inner list is one row of the csv
           all data items are strings; empty cells are empty strings
    """
    try:
        csvfile = open( filename, "w", newline='' )
        filewriter = csv.writer( csvfile, delimiter=",")
        for row in list_of_rows:
            filewriter.writerow( row )
        csvfile.close()

    except:
        print("File", filename, "could not be opened for writing...")


#
# csv_to_html_table_starter
#
#   Shows off how to create an html-formatted string
#   Some newlines are added for human-readability...
#
def csv_to_html_table_starter( csvdata ):
    """ csv_to_html_table_starter
           + an example of a function that returns an html-formatted string
        Run with
           + result = csv_to_html_table_starter( "example_chars.csv" )
        Then run
           + print(result)
        to see the string in a form easy to copy-and-paste...
    """
    # probably should use the readcsv function, above!
    html_string = '<table>\n'    # start with the table tag

    for element in csvdata:
        # print("The element is:", element)
        # print("The first part of the element is:", element[0])
        html_string += '<tr>\n<td>' + element[0] + '</td>\n<td>' + str(element[1]) + '</td>\n</tr>\n'
        # html_string += str(element) + "\n" # "place your table rows and data here!\n" # from list_of_rows !

    html_string += '</table>\n'
    return html_string

# Begin my functions
#
def create_html_page(htmldata, filename):
    """ generates the rest of the text needed for a full html file,
        then puts all the html text together and saves it to a file
    """
    begin = "<html>\n\n<body>\n\n<p>\n"
    end = "\n</p>\n\n</body>\n\n</html>"
    full_text = begin + htmldata + end
    f = open(filename, "w")
    f.write(full_text)
    f.close()



def dict_to_list(dictionary):
    dict_list = []
    for key, value in dictionary.items():
        dict_list.append([key, value])
    return dict_list


def Wcount_first():
    LoR = readcsv("wds.csv")
    counts = defaultdict(int)
    for row in LoR:
        word = str(row[0])
        num = float(row[1])
        first_letter = word[0]
        counts[first_letter] += num
    data = dict_to_list(counts)
    output_html = csv_to_html_table_starter(data)
    create_html_page(output_html, "weighted_count_first.html")


def Wcount_last():
    LoR = readcsv("wds.csv")
    counts = defaultdict(int)
    for row in LoR:
        word = str(row[0])
        num = float(row[1])
        last_letter = word[-1]
        counts[last_letter] += num
    data = dict_to_list(counts)
    output_html = csv_to_html_table_starter(data)
    create_html_page(output_html, "weighted_count_last.html")


def Wcount_middle():
    LoR = readcsv("wds.csv")
    counts = defaultdict(int)
    for row in LoR:
        word = str(row[0])
        num = float(row[1])
        middle_letter = word[int(len(word)/2)-1]
        counts[middle_letter] += num
    data = dict_to_list(counts)
    output_html = csv_to_html_table_starter(data)
    create_html_page(output_html, "weighted_count_middle.html")


def main():
    """ run this file as a script """
    # LoL = readcsv( "wds.csv" )
    # print(LoL[:10])

    # test writing
    # write_to_csv( LoL[:10], "tenrows.csv" )

    # text csv_to_html_table_starter
    # output_html = csv_to_html_table_starter( LoL[:10] )
    # print("\noutput_html is\n\n" + output_html)
    # create_html_page(output_html, "test.html")
    Wcount_first()
    Wcount_last()
    Wcount_middle()


if __name__ == "__main__":
    main()
