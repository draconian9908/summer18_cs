#
# coding: utf-8
#
# hw8pr1.py - the k-means algorithm -- with pixels...
#

# import everything we need...
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
import numpy as np
import utils
import cv2

# choose an image...
# IMAGE_NAME = "./jp.png"  # Jurassic Park
# IMAGE_NAME = "./batman.png"
# IMAGE_NAME = "./hmc.png"
# IMAGE_NAME = "./thematrix.png"
# IMAGE_NAME = "./fox.jpg"
IMAGE_NAME = "hunter.jpg"
image = cv2.imread(IMAGE_NAME, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to be a list of pixels
image_pixels = image.reshape((image.shape[0] * image.shape[1], 3))

K_MEAN = False
if K_MEAN == True:
    # choose k (the number of means) in  NUM_MEANS
    # and cluster the pixel intensities
    NUM_MEANS = [2,3,5,7]
    bars = []
    titles = []
    for num in NUM_MEANS:
        clusters = KMeans(n_clusters = num)
        clusters.fit(image_pixels)

        # After the call to fit, the key information is contained
        # in  clusters.cluster_centers_ :
        # count = 0
        # for center in clusters.cluster_centers_:
        #     print("Center #", count, " == ", center)
        #     # note that the center's values are floats, not ints!
        #     center_integers = [int(p) for p in center]
        #     print("   and as ints:", center_integers)
        #     count += 1

        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = utils.centroid_histogram(clusters)
        bar = utils.plot_colors(hist, clusters.cluster_centers_)
        bars.append(bar)
        title = str(num) + " means"
        titles.append(title)


    # in the first figure window, show our image
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)

    # in the second figure window, show the pixel histograms
    #   this starter code has a single value of k for each
    #   your task is to vary k and show the resulting histograms
    # this also illustrates one way to display multiple images
    # in a 2d layout (fig == figure, ax == axes)
    gs = gridspec.GridSpec(4,2, width_ratios=[2,1])
    ax1 = plt.subplot(gs[:,0]); ax2 = plt.subplot(gs[0,1]); ax3 = plt.subplot(gs[1,1]); ax4 = plt.subplot(gs[2,1]); ax5 = plt.subplot(gs[3,1])
    ax1.imshow(image);    ax1.set_title("Original");   ax1.axis('off')
    ax2.imshow(bars[0]);    ax2.set_title(titles[0]);   ax2.axis('off')
    ax3.imshow(bars[1]);    ax3.set_title(titles[1]);   ax3.axis('off')
    ax4.imshow(bars[2]);    ax4.set_title(titles[2]);   ax4.axis('off')
    ax5.imshow(bars[3]);    ax5.set_title(titles[3]);   ax5.axis('off')
    plt.savefig("kmeans.jpg")
    plt.show()

POSTERIZE = True
if POSTERIZE == True:
    num = 4
    clusters = KMeans(n_clusters = num)
    clusters.fit(image_pixels)
    num_rows, num_cols, num_channels = image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            dist = []
            for center in clusters.cluster_centers_:
                dist.append(np.sqrt(np.sum((image[row,col] - center)**2)))
            dist = np.asarray(dist)
            min_i = np.argmin(dist)
            image[row,col] = clusters.cluster_centers_[min_i]
    plt.imshow(image)
    plt.axis('off')
    plt.show()



#
# comments and reflections on hw8pr1, k-means and pixels
"""
 + Which of the paths did you take:
    + posterizing or
    + algorithm-implementation

 + How did it go?  Which file(s) should we look at?
 + Which function(s) should we try...
"""
#
#
