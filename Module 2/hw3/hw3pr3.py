#
# hw3pr3.py
#
# Visualizing your own data with matplotlib...
#
# Here, you should include functions that produce two visualizations of data
#   of your own choice. Also, include a short description of the data and
#   the visualizations you created. Save them as screenshots or as saved-images,
#   named datavis1.png and datavis2.png in your hw3.zip folder.
#
# Gallery of matplotlib examples:   http://matplotlib.org/gallery.html
#
# List of many large-data sources:    https://docs.google.com/document/d/1dr2_Byi4I6KI7CQUTiMjX0FXRo-M9k6kB2OESd7a2ck/edit
#     and, the birthday data in birth.csv is a reasonable fall-back option, if you'd like to use that...
#          you could create a heatmap or traditional graph of birthday frequency variations over the year...
#

"""
Short description of the two data visualizations...


    +++  Please don't use these two! These are simply placeholders (from xkcd...)  +++

    They do show how to save out a plot to a file with fig.savefig , which is useful.


"""



#
# datavis1()
#
"""
From:  http://matplotlib.org/examples/showcase/xkcd.html
"""

import matplotlib.pyplot as plt
import numpy as np

def datavis1():
    """ run this function for the first data visualization """
    with plt.xkcd():
        # Based on "Stove Ownership" from XKCD by Randall Monroe
        # http://xkcd.com/418/

        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.xticks([])
        plt.yticks([])
        ax.set_ylim([-30, 10])

        data = np.ones(100)
        data[70:] -= np.arange(30)

        plt.annotate(
            'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
            xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

        plt.plot(data)

        plt.xlabel('time')
        plt.ylabel('my overall health')
        fig.text(
            0.5, 0.05,
            '"Stove Ownership" from xkcd by Randall Monroe',
            ha='center')

        # Based on "The Data So Far" from XKCD by Randall Monroe
        # http://xkcd.com/373/

        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        ax.bar([0, 1], [0, 100], 0.25)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([0, 1])
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([0, 110])
        ax.set_xticklabels(['CONFIRMED BY\nEXPERIMENT', 'REFUTED BY\nEXPERIMENT'])
        plt.yticks([])

        plt.title("CLAIMS OF SUPERNATURAL POWERS")

        fig.text(
            0.5, 0.05,
            '"The Data So Far" from xkcd by Randall Monroe',
            ha='center')

    # save to file
    fig.savefig('datavis1.png', bbox_inches='tight')
    # and show it on the screen
    plt.show()

# run it!
# datavis1()


#
# datavis2()
#
"""
From:  http://matplotlib.org/xkcd/examples/pylab_examples/manual_axis.html
"""

import numpy as np
from pylab import figure, show
import matplotlib.lines as lines
import matplotlib.pyplot as plt

def make_xaxis(ax, yloc, offset=0.05, **props):
    """ custom-axis (x) example
    """
    xmin, xmax = ax.get_xlim()
    locs = [loc for loc in ax.xaxis.get_majorticklocs()
            if loc>=xmin and loc<=xmax]
    tickline, = ax.plot(locs, [yloc]*len(locs),linestyle='',
            marker=lines.TICKDOWN, **props)
    axline, = ax.plot([xmin, xmax], [yloc, yloc], **props)
    tickline.set_clip_on(False)
    axline.set_clip_on(False)
    for loc in locs:
        ax.text(loc, yloc-offset, '%1.1f'%loc,
                horizontalalignment='center',
                verticalalignment='top')

def make_yaxis(ax, xloc=0, offset=0.05, **props):
    """ custom-axis (y) example
    """
    ymin, ymax = ax.get_ylim()
    locs = [loc for loc in ax.yaxis.get_majorticklocs()
            if loc>=ymin and loc<=ymax]
    tickline, = ax.plot([xloc]*len(locs), locs, linestyle='',
            marker=lines.TICKLEFT, **props)
    axline, = ax.plot([xloc, xloc], [ymin, ymax], **props)
    tickline.set_clip_on(False)
    axline.set_clip_on(False)

    for loc in locs:
        ax.text(xloc-offset, loc, '%1.1f'%loc,
                verticalalignment='center',
                horizontalalignment='right')

def datavis2():
    """ run this function for the second data visualization """
    with plt.xkcd():
        props = dict(color='black', linewidth=2, markeredgewidth=2)
        x = np.arange(200.)
        y = np.sin(2*np.pi*x/200.) + np.random.rand(200)-0.5
        fig = figure(facecolor='white')
        ax = fig.add_subplot(111, frame_on=False)
        ax.axison = False
        ax.plot(x, y, 'd', markersize=8, markerfacecolor='blue')
        ax.set_xlim(0, 200)
        ax.set_ylim(-1.5, 1.5)
        make_xaxis(ax, 0, offset=0.1, **props)
        make_yaxis(ax, 0, offset=5, **props)
        # save to file
        fig.savefig('datavis2.png', bbox_inches='tight')
        # and show it on the screen
        show()

# run it!
# datavis2()



# Begin my functions


def readcsv(filename):
    try:
        csvfile = open( filename, newline='' )
        csvrows = csv.reader( csvfile )

        all_rows = []
        for row in csvrows:
            all_rows.append( row )

        del csvrows
        csvfile.close()
        return all_rows

    except FileNotFoundError as e:
        print("File not found: ", e)
        return []



if __name__ == "__main__":
    main()
