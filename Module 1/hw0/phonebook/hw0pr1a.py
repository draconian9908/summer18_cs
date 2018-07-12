#
# hw0pr1.py
#

# An example function
#
def plus1( N ):
    """ returns a number one larger than its input """
    return N+1



# An example loop (just with a printed countdown)
#
import time

def countdown( N ):
    """ counts downward from N to 0 printing only """
    for i in range(N,-1,-1):
        print("i ==", i)
        time.sleep(0.01)

    return    # no return value here!

# Beginning work:

def times42( s ):
    """ prints string `s` 42 times on separate lines """
    print((s + "\n") * 42)
    return # No return value

def alien( N ):
    """ return string `alien` with exactly N i's """
    return "al" + ("i" * N) + "en"

def count_digits( s ):
    """ returns number of digits in string `s` """
    if s.isalpha():
        return 0
    elif s.isdigit():
        return len(s)
    else:
        count = 0
        for n in range(11):
            c = s.count(str(n))
            count += c
        return count

def clean_digits( s ):
    """ returns only the digits in string `s` """
    storage = []
    for element in s:
        if element.isdigit():
            storage.append(element)
    return "".join(storage)

def clean_word( s ):
    """ returns only the letters of string `s`, all lowercase """
    storage = []
    for element in s:
        if element.isalpha():
            storage.append(element)
    return "".join(storage).lower()

# edited version for use in pr2
def clean_name( s ):
    """ returns only the letters of string `s`, all lowercase """
    storage = []
    for element in s:
        if element.isalpha() or element == "," or element == " ":
            storage.append(element)
    return "".join(storage)

# An example main() function - to keep everything organized!
#
def main():
    """ main function for organizing -- and printing -- everything """

    # sign on
    print("\n\nStart of main()\n\n")

    # testing plus1
    result = plus1( 41 )
    print("plus1(41) returns", result)

    # testing countdown
    print("Testing countdown(5):")
    countdown(5)  # should print things -- with dramatic pauses!

    # testing times42
    print("Testing times42(42):")
    times42("42") # should print `42` on 42 lines

    # testing alien
    result = alien(42)
    print("alien(42) returns", result)

    # testing count_digits
    result = count_digits("shdu34n678")
    print("count_digits('shdu34n678') returns", result)

    # test clean_digits
    result = clean_digits("shdu34n678")
    print("clean_digits('shdu34n678') returns", result)

    # test clean_word
    result = clean_word("shdu34n678")
    print("clean_word('shdu34n678') returns", result)

    # test clean_name
    result = clean_name("345-6798Rocker, Norma")
    print("clean_name('345-6798Rocker, Norma') returns", result)

    # sign off
    print("\n\nEnd of main()\n\n")



# This conditional will run main() when this file is executed:
#
if __name__ == "__main__":
    main()



# ++ The challenges:  Create and test as many of these five functions as you can.
#
# The final three may be helpful later...
#
# times42( s ):      which should print the string s 42 times (on separate lines)
# alien( N ):          should return the string "aliii...iiien" with exactly N "i"s
# count_digits( s ):    returns the number of digits in the input string s
# clean_digits( s ):    returns only the digits in the input string s
# clean_word( s ):    returns an all-lowercase, all-letter version of the input string s
