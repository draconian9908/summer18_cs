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

