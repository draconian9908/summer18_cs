# coding: utf-8

"""
#
# Here is an example of using the Web's Wisdom to answer the question
#
#     Who will win the superbowl next weekend... ?
#
#         Note: not limited to next weekend's teams!
#         Another note: gambling on future outcomes using this technique _not_ recommended!
# 
#     use/alter main() to find out the results of our predictions!  :-)
#
"""
# 2016:
# team_1 = 'carolina-panthers'  
# team_2 = 'denver-broncos'     

# 2017:
# team_1 = 'new-england-patriots'
# team_2 = 'atlanta-falcons'

# 2018:
# team_1 = 'new-england-patriots'
# team_2 = 'philadelphia-eagles'

# 
# Here is our "web scavenging" approach to the winner and the score:
#
# 1. first we grab the pages that define each team's colors
# 2. then, use bs4 to parse those pages and return a list of colors
# 3. then, grab the page that defines the popularity of team colors
# 4. finally, use bs4 to compute a score for each team based on its colors
#

import requests
from bs4 import BeautifulSoup

#
# get_color_page()
#
def get_color_page():
    """ This function requests the most-popular-colors page
        and parses it with Beautiful Soup, returning the resulting
        Beautiful Soup object, soup
    """
    color_popularity_url = "http://www.thetoptens.com/top-ten-favorite-colors/"
    response = requests.get(color_popularity_url)   # request the page

    if response.status_code == 404:                 # page not found
        print("There was a problem with getting the page:")
        print(color_popularity_url)

    data_from_url = response.text                   # the HTML text from the page
    soup = BeautifulSoup(data_from_url,"lxml")      # parsed with Beautiful Soup
    return soup



#
# find_color_score( color, soup )
#
def find_color_score( color_name, soup ):
    """ find_color_score takes in color_name (a string represnting a color)
        and soup, a Beautiful Soup object returned from a successful run of
        get_color_page
        
        find_color_score returns our predictive model's number of points in
        a potential match up involving a team with that color
        
        the number of points is 21 - ranking, where ranking is from 1 (most
        popular color) to 20 (least popular color) or 21, representing all
        of the others
    """
    ListOfDivs = soup.findAll('div', {'class':"i"})   # the class name happens to be 'i' here...

    for div in ListOfDivs:
        # print(div.em, div.b)                    # checking the subtags named em and b
        this_divs_color = div.b.text.lower()      # getting the text from them (lowercase)
        this_divs_ranking_as_str = div.em.text    # this is the _string_ of the ranking
        
        this_divs_ranking = 21                    # the deafult (integer) ranking: 21
        try:
            this_divs_ranking = int(div.em.text)  # try to convert it to an integer
        except:                                   # if it fails
            pass                                  # do nothing and leave it at 21
        
        if color_name == this_divs_color:         # check if we need to return this one
            return this_divs_ranking
        
    # if we ran through the whole for loop without finding a match, the ranking is 21
    return 21


#
# get_team_colors_page(team)
#
def get_team_colors_page(team):
    """ get_team_colors_page takes in a string with a "city-mascot" formatted team name
        such as atlanta-falcons or carolina-panthers or philadelphia-eagles
        
        it tried to request the appropriate page from teamcolorcodes.com and parse
        it with Beautiful Soup - and it should return that soup object
    """
    team_color_url = "http://teamcolorcodes.com/" + team + "-color-codes/"
    response = requests.get(team_color_url)         # request the page
    
    if response.status_code == 404:                 # page not found
        print("For the team", team, "There was a problem with getting the page")
        print(team_color_url)
        
    data_from_url = response.text                   # the HTML text from the page
    soup = BeautifulSoup(data_from_url,"lxml")      # parsed with Beautiful Soup
    return soup



#
# extract_team_colors( team )
#
def extract_team_colors(soup):
    """ extract_team_colors takes in a beautiful soup object, soup
        and uses Beautiful Soup to extract a list of all of that team's colors
        
        it return that list of colors
        
        (Note that for different teams, you'll need to run the get_team_colors_page
         to obtain soup objects for each page.)
    """
    AllDivs = soup.findAll('div', {'class':"colorblock"})   # find all divs of the right class
    
    list_of_team_colors = []
    for div in AllDivs:
        # LoClasses = div.get('class') # get the list of classes for this tag
        # print("  ",div.text)   # debugging line (one of many...)
        # example: RedHex Color: #cc122cRGB: (204,18,44)CMYK: (13,100,93,4)
        if 'Hex' not in div.text: continue  # doens't match the above model, skip it
        Words = div.text.split('Hex')  
        # splits around 'Hex':  ['Red', 'Color: #cc122cRGB: (204,18,44)CMYK: (13,100,93,4)']
        color = Words[0]  # the word _before_ 'Hex'
        Words = color.split()    # check if there are more than one words in the color
        if len(Words) > 1: color = Words[-1]  # if so, the color is the final word!
        # e.g. midnight green -> green
        color = color.lower()         # make lower case
        list_of_team_colors.append( color )
        if len(list_of_team_colors) > 2: break  # use at most three colors!

    # That's it - return the list (hopefully there are some colors!)
    return list_of_team_colors



#
# put it all together!
#
def main():
    """
    # Here is an example of using the Web's Wisdom to answer the question
    #
    #     Who will win the superbowl next weekend... ?
    #
    #     Not limited to last weekend's teams!
    """
    # 2016
    # team_1 = 'carolina-panthers'  
    # team_2 = 'denver-broncos'    

    # 2017:
    # team_1 = 'new-england-patriots'
    # team_2 = 'atlanta-falcons'

    # 2018:
    team_1 = 'new-england-patriots'
    team_2 = 'philadelphia-eagles'
    # team_2 = 'pittsburgh-steelers'

    # 
    # Here is our "web scavenging" approach:
    #
    # 1. first we grab the pages that define each team's colors
    # 2. then, use bs4 to parse those pages and return a list of colors
    # 3. then, grab the page that defines the popularity of team colors
    # 4. finally, use bs4 to compute a score for each team based on its colors
    #

    # we get the team colors page for each team
    # and we return a BeautifulSoup "soup" object for each!
    team_soup_1 = get_team_colors_page(team_1)
    team_soup_2 = get_team_colors_page(team_2)
    print("Done scraping the team colors.\n")

    # We have a function that actually grabs the colors from the page...
    team_colors_1 = extract_team_colors( team_soup_1 )
    team_colors_2 = extract_team_colors( team_soup_2 )
    print("Team 1 (" + team_1 + ") colors:", team_colors_1)
    print("Team 2 (" + team_2 + ") colors:", team_colors_2)


    # Next, we grab the color-popularity page (and parse it into
    # a BeautifulSoup object...
    # 
    color_popularity_soup = get_color_page()
    print("\nDone scraping the color-popularity page.\n")

    # Finally, we convert the team colors into total scores
    # which will reveal our predicted result
    # Admittedly, our "points" are simply the ranking of how popular a color is.

    # let's use a list comprehension as a reminder of how those work...
    team_1_scores = [ find_color_score(clr, color_popularity_soup) for clr in team_colors_1 ]
    team_2_scores = [ find_color_score(clr, color_popularity_soup) for clr in team_colors_2 ]
    print("Team 1 (" + team_1 + ") scores:", team_1_scores)
    print("Team 2 (" + team_2 + ") scores:", team_2_scores)
    print()
    print("Team 1 (" + team_1 + ") predicted final score:", sum(team_1_scores))
    print("Team 2 (" + team_2 + ") predicted score:", sum(team_2_scores))


    # that's it!
    return

#
# our conditional, kicking off all execution...
#
if __name__ == "__main__":
    main()  # hike!

