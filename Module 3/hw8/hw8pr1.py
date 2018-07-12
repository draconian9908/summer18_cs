# coding: utf-8

# # Introduction to OpenCV - solutions for hw8 (cs35 spring 17)

# Check that you have opencv:
# import cv2

# if this doesn't work, try
"""
[Mac instructions]    At the command prompt (not within python), run
conda install -c menpo opencv3
conda update hdf5

[Windows instructions]     At the command prompt (not within python), run
conda install -c menpo opencv3
"""

# documentation for OpenCV:   http://docs.opencv.org/3.1.0/
# tutorials are in many places, e.g., http://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html


# other imports!
import numpy as np
from matplotlib import pyplot as plt
import cv2

# run %matplotlib for smoother image-display...




# ### Images as pictures: reading, writing, and resize

def opencv_test1():
    """ some initial trials... read and show images """
    # options include cv2.IMREAD_COLOR # color, with no alpha channel
    # also cv2.IMREAD_GRAYSCALE (== 0)
    # or cv2.IMREAD_UNCHANGED (includes alpha)

    # Reading and color-converting an image to RGB
    raw_image = cv2.imread('messi5.jpg',cv2.IMREAD_COLOR)
    #raw_image = cv2.imread('monalisa.jpg',cv2.IMREAD_COLOR)

    # convert an OpenCV image (BGR) to an "ordinary" image (RGB)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)    # plt.show()
    plt.xticks([]), plt.yticks([])  # to hide axis labels
    input("Hit enter to continue...")

    # In[ ]:

    # let's get the flag image next
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR)
    flag = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(flag)
    print("flag.shape is", flag.shape)
    input("Hit enter to continue...")

    return flag   # for use further downstream...




def opencv_test2():
    """ a second set of tests... resize and image and write to file... """
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR)
    flag = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(flag)
    print("flag.shape is", flag.shape)
    input("Hit enter to continue...")

    # want to resize a new image - no problem:
    small_flag = cv2.resize(flag, dsize=(50,50), interpolation=cv2.INTER_LINEAR)
    plt.imshow(small_flag)
    plt.show()
    print("flag.shape is", small_flag.shape)
    input("Hit enter to continue...")
    # there are three resize algorithms (cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC), described here:
    # http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    raw_image = cv2.cvtColor(small_flag, cv2.COLOR_RGB2BGR) # convert back!
    cv2.imwrite( "small_flag.png", raw_image )
    print("The file small_flag.png was written...")
    print("Also returning that image...")

    return small_flag






def opencv_test3():
    """ a third set of tests... reread image and access its pixels and channels... """

    # re-read to check the small image:
    raw_image = cv2.imread('small_flag.png',cv2.IMREAD_COLOR)
    small_flag = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(small_flag)
    input("Hit enter to continue...")


    # ### Images as pixels: accessing their individual components
    print("Here are the first four pixels in the top row:")
    print(small_flag[0,0]) # row, col == 0,0  r,g,b == 37, 138, 175
    print("Its r, g, and b components:", small_flag[0,0,0], ",", small_flag[0,0,1], ",", small_flag[0,0,2])
    print(small_flag[0,1]) # row, col
    print(small_flag[0,2]) # row, col
    print(small_flag[0,3]) # row, col

    if True:
        print("printing all pixels of the small_flag image")
        NUMROWS, NUMCOLS, NUMCOLORS  = small_flag.shape
        for r in range(NUMROWS):
            for c in range(NUMCOLS):
                print(small_flag[r,c], end=" ")
            print()
        print("end of printing all pixels")

    return small_flag   # for downstream processing...








def opencv_test4():
    """ an example of a single-image filter
        a filter is an image-in -> image-out transformation
    """
    # blur - a common photoshop filter...
    raw_image = cv2.imread('spam.png',cv2.IMREAD_COLOR)
    spam = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(spam)
    plt.title("Unblurred")

    blurred_spam = cv2.blur( spam, ksize=(5,5) )  # average of 5x5 box (each direction)
    plt.figure()
    plt.imshow(blurred_spam)
    plt.title("Blurred")







def opencv_test5():
    """ more image-filter examples """
    # but we want to be able to go _beyond_ Photoshop (as needed...)
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    #image_r, image_g, image_b = cv2.split( image )  # the three channels OR
    image_r = image[:,:,0]  # red
    image_g = image[:,:,1]  # gre
    image_b = image[:,:,2]  # blu

    # here is a 1d layout of images:
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    ax1[0].imshow(image);
    ax1[0].set_title("orig");
    ax1[0].axis('off')
    ax1[1].imshow(image_b)#, cmap='gray');
    ax1[1].set_title("blue channel alone");
    ax1[1].axis('off')
    plt.show(fig1)

    # here is a 2d layout of grayscale images:
    image_r = image[:,:,0]  # red
    image_g = image[:,:,1]  # gre
    image_b = image[:,:,2]  # blu

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
    ax[0,0].imshow(image);  #
    ax[0,0].set_title("orig")
    ax[0,1].imshow(image_b, cmap='Blues')#, cmap='gray');  #
    ax[0,1].set_title("blue")
    ax[1,0].imshow(image_r, cmap='Reds')#, cmap='gray');  #
    ax[1,0].set_title("red")
    ax[1,1].imshow(image_g, cmap='Greens')#, cmap='gray');  #
    ax[1,1].set_title("green")
    for row in range(2):
        for col in range(2):
            ax[row,col].axis('off')
    plt.show(fig)




def opencv_test6():
    """ more image-filter examples - base yours from these examples... """
    # #### an example of a color conversion in OpenCV
    # but we want to be able to go _beyond_ Photoshop (as needed...)
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    # color conversions are here:
    # http://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#gsc.tab=0
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Grayscale")
    plt.show()
    input("Hit return to continue...")

    # what will happen here?
    gray2 = gray_image//2
    plt.imshow(gray2, cmap='gray', vmin=0, vmax=255)
    plt.title("HALF grayscale")
    plt.show()
    input("Hit return to continue...")

    # example of thresholding!
    # numpy arrays are cleverer than plain lists...
    a = np.array( [ [1,2,3], [4,5,6], [7,8,9] ] )
    print(a)
    indices = a < 4.5
    print(indices)
    a[indices] = 0
    print(a)

    # types are important!
    # numpy types: http://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html
    a = a.astype(float)
    print(a)
    a = a.astype(int)
    print(a)


    # #### let's try with images
    # here are some thresholded images
    image_gt142_indices = gray_image > 142
    image_gt142 = gray_image.copy()
    image_gt142[image_gt142_indices] = 255
    image_gt142[np.logical_not(image_gt142_indices)] = 0
    # note that this shows how to take logical operators of index arrays...

    plt.imshow(image_gt142, cmap='gray', vmin=0, vmax=255)
    plt.title("Thresholded at gray level >= 142")
    input("Hit return to continue...")

    # here are some other thresholded images
    image_lt42_indices = gray_image < 42
    image_lt42 = image.copy()
    image_lt42[image_lt42_indices] = 255
    image_lt42[np.logical_not(image_lt42_indices)] = 0
    plt.imshow(image_lt42, cmap='gray', vmin=0, vmax=255)
    plt.title("Thresholded at gray level <= 42")
    plt.show()

    image_not_btw_42_142_indices =   np.logical_or(image_gt142_indices,image_lt42_indices)
    image_btw_42_142 = image.copy()
    image_btw_42_142[image_not_btw_42_142_indices] = 0
    image_btw_42_142[np.logical_not(image_not_btw_42_142_indices)] = 255
    plt.imshow(image_btw_42_142, cmap='gray', vmin=0, vmax=255)
    plt.title("Thresholded at gray level >= 42 & <= 142")
    plt.show()

    # here is a 2d layout of images:
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
    ax[0,0].imshow(image, cmap='gray');  #ax[0,0].set_title("orig")
    ax[0,0].set_title('orig')
    ax[0,1].imshow(image_gt142, cmap='gray');  #ax[1,1].set_title("blue")
    ax[0,1].set_title('greater than 142')
    ax[1,0].imshow(image_btw_42_142, cmap='gray');  #ax[1,0].set_title("green")
    ax[1,0].set_title('between 42 and 142')
    ax[1,1].imshow(image_lt42, cmap='gray');  #ax[0,1].set_title("red")
    ax[1,1].set_title('less than 42')

    for row in range(2):
        for col in range(2):
            ax[row,col].axis('off')
    plt.show()


#
# Here are several example filters on which to base your own...
#
def example_filter( image ):
    """ an example of a pixel-by-pixel filter
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [b, b, b]  # [0,0,b]
    return new_image


from matplotlib import pyplot as plt
#
# try_filter!
#
def try_filter( image1 ):
    """ try out a filter! """
    from matplotlib import pyplot as plt
    new_image = two_image_filter( image1,image2 )
    plt.imshow(new_image)
    plt.show()


# Problem #1 - a few example filters
#
# The starter file provides these -- your task: to create two additional filters: one
# that transforms a single image; one that composited two images together (in some way)
#



# for loops over the image
def example_filter_bgr( image ):
    """ an example of a pixel-by-pixel filter
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [b, g, r]
    return new_image


def example_filter_gbr( image ):
    """ an example of a pixel-by-pixel filter
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [g, b, r]
    return new_image


def example_filter_inv( image ):
    """ an example of a pixel-by-pixel filter
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [255-r, 255-g, 255-b]
    return new_image

def example_filter_ig( image ):
    """ an example of a pixel-by-pixel filter
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [r, 255-g, b]
    return new_image

def example_filter_delbot2( image ):
    """ an example of a pixel-by-pixel filter
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [r << 2, g << 2, b << 2]
    return new_image

def example_filter_deltop2( image ):
    """ an example of a pixel-by-pixel filter
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [r>>2 , g>>2, b>>2 ]
    return new_image




# ## Problem 1:  write-your-own-filter
#
# Part A will be at least one filter that takes in a single image, transforms its pixels in an unusual way (something not in Photoshop's menu options, perhaps?), and then runs that filter on at least two of your own images.
#
# Part B will ask you to create at least one filter -- but one that combines _two_ images, meshing their pixels in a creative way, and outputting a single image. It's OK to resize the images so that they;re the same shape.
# use the above examples as a starting point...



def new_filter( image ):
    """ thresholds each color individually at half of 255:
        less than 255 -> that color for that pixel becomes 0
        greater than 255 -> that color for that pixel becomes 255
    """
    image_r, image_g, image_b = cv2.split(image)
    hn = 255.0/2
    r_indices_h = image_r > hn
    r_indices_l = image_r < hn
    g_indices_h = image_g > hn
    g_indices_l = image_g < hn
    b_indices_h = image_b > hn
    b_indices_l = image_b < hn
    image_r[r_indices_h] = 255
    image_r[r_indices_l] = 0
    image_g[g_indices_h] = 255
    image_g[g_indices_l] = 0
    image_b[b_indices_h] = 255
    image_b[b_indices_l] = 0
    new_image = cv2.merge([image_r,image_g,image_b])
    return new_image



def two_image_filter( image1, image2 ):
    """ better docstring needed! Explain your two-image filter... """
    sr = min(image1.shape[0], image2.shape[0])
    sc = min(image1.shape[1], image2.shape[1])
    im1 = cv2.resize(image1, None, fx=sc/image1.shape[1], fy=sr/image1.shape[0], interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(image2, None, fx=sc/image2.shape[1], fy=sr/image2.shape[0], interpolation=cv2.INTER_AREA)
    im1_r, im1_g, im1_b = cv2.split(im1)
    im2_r, im2_g, im2_b = cv2.split(im2)
    r_indices = np.greater(im2_r, im1_r)
    g_indices = np.greater(im2_g, im1_g)
    b_indices = np.greater(im2_b, im1_b)
    im1_r[r_indices] = im2_r[r_indices]
    im1_g[g_indices] = im2_g[g_indices]
    im1_b[b_indices] = im2_b[b_indices]
    new_image = cv2.merge([im1_r,im1_g,im1_b])
    return new_image





def main():
    raw_image1 = cv2.imread('brindle_bull_terrier2.jpg',cv2.IMREAD_COLOR)
    raw_image2 = cv2.imread('neural_landscape.jpg',cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(raw_image1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(raw_image2, cv2.COLOR_BGR2RGB)
    # plt.imshow(im)
    # plt.show()
    try_filter(im1,im2)

if __name__ == "__main__":
    main()
