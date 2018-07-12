#
# hw2pr1.py - write-your-own-web-engine...
#
# then, improve the page's content and styling!
#


import re
from copy import deepcopy


def apply_headers( OriginalLines ):
    """ should apply headers, h1-h5, as tags
    """
    # loop for all headings: h1-h5
    NewLines =[]
    for line in OriginalLines:
        if line.startswith("#####"):
            line = "<h5>" + line[5:] + "</h5>"
        elif line.startswith("####"):
            line = "<h4>" + line[4:] + "</h4>"
        elif line.startswith("###"):
            line = "<h3>" + line[3:] + "</h3>"
        elif line.startswith("##"):
            line = "<h2>" + line[2:] + "</h2>"
        elif line.startswith("#"):
            line = "<h1>" + line[1:] + "</h1>"
        NewLines += [ line ]
    return NewLines

def apply_wordstyling( OriginalLines ):
    """ should apply wordstyling here...
    """
    # loop for the word-stylings: here, ~word~
    NewLines =[]
    for line in OriginalLines:
        # regular expression example!
        line = re.sub(r"~(.*)~", r"<i>\1</i>", line)
        line = re.sub(r"[_](.*)[_]", r"<u>\1</u>", line)
        line = re.sub(r"[*](.*)[*]", r"<b>\1</b>", line)
        line = re.sub(r"[@](.*)[@]", r"<a href=\1>Link</a>", line)
        line = re.sub(r"[&](.*)[&]", r"<div id=animation class=blink>\1</div>", line)
        # let's practice some others...!
        # regular expressions:  https://docs.python.org/3.4/library/re.html
        NewLines += [ line ]
    return NewLines


def listify(OriginalLines):
    """ convert lists beginning with "   +" into HTML """
    NewLines = []
    # loop for lists
    previous = 0
    for line in OriginalLines:
        if line.startswith("   +") and previous == 0:
            previous = 1
            line = "<ul>\n<li>" + line[4:] + "</li>"
        elif line.startswith("   +") and previous == 1:
            previous = 1
            line = "<li>" + line[4:] + "</li>"
        elif not line.startswith("   +") and previous == 1:
            previous = 0
            line = "</ul>\n" + line
        NewLines += [ line ]
    return NewLines



def main():
    """ handles the conversion from the human-typed file to the HTML output """

    HUMAN_FILENAME = "starter.txt"
    OUTPUT_FILENAME = "starter.html"

    f = open(HUMAN_FILENAME, "r", encoding="latin1")
    contents = f.read()
    f.close()

    print("Original contents were\n", contents, "\n")

    OriginalLines = contents.split("\n")  # split to create a list of lines
    NewLines = apply_headers( OriginalLines )
    NewLines = apply_wordstyling(NewLines)
    NewLines = listify(NewLines)

    # finally, we join everything with newlines...
    final_html = '\n'.join(NewLines)

    print("\nFinal contents are\n", final_html, "\n")

    f = open(OUTPUT_FILENAME, "w")     # write this out to a file...
    f.write( final_html )
    f.close()
    # then, render in your browser...


if __name__ == "__main__":
    main()
