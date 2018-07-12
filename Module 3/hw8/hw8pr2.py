# ## Problem 2:  green-screening!
#
# This question asks you to write one function that takes in two images:
#  + orig_image  (the green-screened image)
#  + new_bg_image (the new background image)
#
# It also takes in a 2-tuple (corner = (0,0)) to indicate where to place the upper-left
#   corner of orig_image relative to new_bg_image
#
# The challenge is to overlay the images -- but only the non-green pixels of
#   orig_image...
#

#
# Again, you'll want to borrow from hw7pr1 for
#  + opening the files
#  + reading the pixels
#  + create some helper functions
#    + defining whether a pixel is green is the key helper function to write!
#  + then, creating an output image (start with a copy of new_bg_image!)
#
# Happy green-screening, everyone! Include at least TWO examples of a background!
#

import numpy as np
from matplotlib import pyplot as plt
import cv2

def read_images(file1, file2):
    """ reads the images from the files and converts them to image objects """
    raw_image1 = cv2.imread(file1,cv2.IMREAD_COLOR)
    raw_image2 = cv2.imread(file2,cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(raw_image1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(raw_image2, cv2.COLOR_BGR2RGB)
    return [im1, im2]

def fix_size(orig_image, new_bg_image):
    """ matches the sizes of both images to the smallest dimensions between them """
    sr = min(orig_image.shape[0], new_bg_image.shape[0])
    sc = min(orig_image.shape[1], new_bg_image.shape[1])
    im1 = cv2.resize(orig_image, None, fx=sc/orig_image.shape[1], fy=sr/orig_image.shape[0], interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(new_bg_image, None, fx=sc/new_bg_image.shape[1], fy=sr/new_bg_image.shape[0], interpolation=cv2.INTER_AREA)
    return [im1, im2]

def find_green(orig_im):
    check_num = 20
    num_rows, num_cols, num_chans = orig_im.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = orig_im[row,col]
            if g > r+check_num and g > b+check_num:
            # if r <= 80 and b <= 80 and g >= 100:
                orig_im[row,col] = [0,255,0]
    return orig_im

def write_image(new_image):
    writing_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR) # convert back!
    cv2.imwrite( "green_screen_head.png", writing_image)

# Here is a signature for the green-screening...
# remember - you will want helper functions!
def green_screen( orig_image, new_bg_image, corner=(0,0) ):
    """ be sure to include a better docstring here! """
    orig_im, new_bg_im = fix_size(orig_image, new_bg_image)
    orig_im = find_green(orig_im)
    o_r, o_g, o_b = cv2.split(orig_im)
    n_r, n_g, n_b = cv2.split(new_bg_im)
    for i in range(o_r.shape[0]):
        for j in range(o_r.shape[1]):
            # print(o_g[i,j])
            if o_r[i,j] != 0 and o_b[i,j] != 0 and o_g[i,j] != 255:
                n_r[i,j] = o_r[i,j]
                n_g[i,j] = o_g[i,j]
                n_b[i,j] = o_b[i,j]
    new_image = cv2.merge([n_r, n_g, n_b])
    write_image(new_image)
    return new_image




def main(file1,file2):
    orig_image, new_bg_image = read_images(file1,file2)
    new_image = green_screen(orig_image, new_bg_image)
    plt.imshow(new_image)
    plt.show()


if __name__ == "__main__":
    main('green_head.jpg','neural_landscape.jpg')
