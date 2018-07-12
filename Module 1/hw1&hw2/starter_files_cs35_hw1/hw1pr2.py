#
# starting examples for cs35, week2 "Web as Input"
#

import requests
import string
import json


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Problem 2 starter code
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#
#
#

def apple_api(artist_name):
    """ searches Apple's iTunes API using an artists name
        to find their artist ID on iTunes
    """
    ### Use the search url to get an artist's itunes ID
    search_url = "https://itunes.apple.com/search"
    parameters = {"term":artist_name,"entity":"musicArtist","media":"music","limit":200}
    result = requests.get(search_url, params=parameters)
    data = result.json()

    # save to a local file so we can examine it
    filename_to_save = "appledata.json"
    f = open( filename_to_save, "w" )     # opens the file for writing
    string_data = json.dumps( data, indent=2 )  # this writes it to a string
    f.write(string_data)                        # then, writes that string to a file...
    f.close()                                   # and closes the file
    print("\nfile", filename_to_save, "written.")

    # Find and return the artist ID
    f = open(filename_to_save, "r")
    contents = f.read()
    i = contents.find("artistId")
    j = contents.find(" ", i)
    k = contents.find(",", j)
    artistid = contents[j+1:k]

    return artistid


#
#
#
def apple_api_lookup(artistId):
    """
    Takes an artistId and grabs a full set of that artist's albums.
    "The Beatles"  has an id of 136975
    "Kendrick Lamar"  has an id of 368183298
    "Taylor Swift"  has an id of 159260351

    Then saves the results to the file "appledata_full.json"

    This function is complete, though you'll likely have to modify it
    to write more_productive( , ) ...
    """
    lookup_url = "https://itunes.apple.com/lookup"
    parameters = {"entity":"album","id":artistId}
    result = requests.get(lookup_url, params=parameters)
    data = result.json()

    # save to a file to examine it...
    filename_to_save="appledata_full.json"
    f = open( filename_to_save, "w" )     # opens the file for writing
    string_data = json.dumps( data, indent=2 )  # this writes it to a string
    f.write(string_data)                        # then, writes that string to a file...
    f.close()                                   # and closes the file
    print("\nfile", filename_to_save, "written.")

    # we'll leave the processing to another function...
    return



#
#
#
def apple_api_lookup_process():
    """ example opening and accessing a large appledata_full.json file...
        You'll likely want to do more!
    """
    filename_to_read="appledata_full.json"
    f = open( filename_to_read, "r" )
    string_data = f.read()
    data = json.loads( string_data )
    # print("the raw json data is\n\n", data, "\n")

    # for live investigation, here's the full data structure
    return data

# Begin my functions:
#
def num_songs(data):
    """ takes the parsed json data and counts the total number of num_songs
        from the search results (counts number of tracks in all albums)
    """
    all_tracks = []
    results = data.get("results")
    for result in results[1:]:
        track_count = result.get("trackCount")
        all_tracks.append(int(track_count))
    return sum(all_tracks)


def more_productive(artist1,artist2):
    """ takes two artist names as strings, finds their IDs on iTunes,
        then counts the total number of songs for both of them
    """
    search_url = "https://itunes.apple.com/search"
    # Search for artist1
    artistid1 = apple_api(artist1)
    artistid2 = apple_api(artist2)

    apple_api_lookup(artistid1)
    data1 = apple_api_lookup_process()

    apple_api_lookup(artistid2)
    data2 = apple_api_lookup_process()

    num1 = num_songs(data1)
    num2 = num_songs(data2)
    print("\nThe total number of songs in iTunes:\nFor", artist1, ":", num1,
            "\nFor", artist2, ":", num2)



#
# main()  for testing problem 2's functions...
#
def main():
    """ a top-level function for testing things... """
    # routine for getting the artistId

    # artistId = apple_api("The Beatles") # should return 136975
    # artistId = apple_api("Kendrick Lamar") # should return 368183298
    artistId = apple_api("Taylor Swift") # should return 159260351
    print("artistId is", artistId)

    apple_api_lookup(artistId)
    data = apple_api_lookup_process()
    number = num_songs(data)
    print("The total number of songs on iTunes by 'Taylor Swift' is", number)

    more_productive( "Katy Perry", "Steve Perry" )
    # get each one's id
    # get each one's file
    # compare number of albums! Done!
    # then ask two of your own questions


#
# passing the mic (of control) over to Python here...
#
if __name__ == "__main__":
    main()
