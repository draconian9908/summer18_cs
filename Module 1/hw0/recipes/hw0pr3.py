#
# hw0pr3.py ~ recipe analysis
#
# Name(s): Lydia
#

#
# be sure your file runs from this location,
# relative to the "recipes" files and directories
#

import os
import shutil

sweet = []
savory = []
vegetarian = []

# Beginning work
def is_sweet_savory(fullfile):
    """ goes through title area to determine
        if the recipe is for a sweet or savory pie,
        then puts the full filename in the corresponding list
    """
    global sweet
    global savory
    f = open(fullfile)
    contents = f.read()
    title = contents[:51]
    if title.find("Sweet") > 0:
        sweet.append(fullfile)
    if title.find("Savory") > 0:
        savory.append(fullfile)


def is_vegetarian(fullfile):
    """ goes through ingredients to determine
        if there's meat in the pie,
        puts full filename in vegetarian list if not
    """
    global vegetarian
    f = open(fullfile)
    contents = f.read()
    i = contents.index("Ingredients:")
    j = contents.index("Instructions:")
    ingredients = contents[i+12:j]
    if ingredients.find("beef") < 0 and ingredients.find("chicken") < 0 and ingredients.find("pork") < 0:
        vegetarian.append(fullfile)


def organize_files(path):
    """ walks through an entire directory structure
        and uses is_sweet_savory and is_vegetarian
        to organize files into appropriate lists
    """

    AllFiles = list(os.walk(path))
    old_files = []
    folders = []

    for item in AllFiles:
        foldername, LoDirs, LoFiles = item
        folders.append(foldername)
        for file in LoFiles:
            fullfile = foldername + "/" + file
            old_files.append(fullfile)
            is_sweet_savory(fullfile)
            is_vegetarian(fullfile)

    # print("Sweet Recipes:\n", sweet)
    # print("Veg Recipes:\n", vegetarian)

    for recipe in vegetarian:
        if recipe[-2:] == "py":
            vegetarian.remove(recipe)

    return [old_files, folders]


def organize_directory():
    """ takes sweet, savory, and vegetarian lists of file names
        and creates directories for them and moves the files there,
        then deletes empty folders
    """
    global sweet
    global savory
    global vegetarian

    os.mkdir("./savory_recipes")
    os.mkdir("./sweet_recipes")
    os.mkdir("./savory_recipes/vegetarian_recipes")

    for recipe in sweet:
        shutil.copy(recipe,"./sweet_recipes")
    for recipe in savory:
        shutil.copy(recipe, "./savory_recipes")
    for recipe in vegetarian:
        shutil.copy(recipe, "./savory_recipes/vegetarian_recipes")



# __main__ function
def main():
    """ overall function to run all examples """

    print("Start of main()\n")

    old_files, folders = organize_files(".")
    organize_directory()

    for file in old_files:
        if file[-2:] != "py":
            os.remove(file)
    for folder in folders:
        if len(folder) > 1:
            os.removedirs(folder)

    print("End of main()\n")


if __name__ == "__main__":
    main()
