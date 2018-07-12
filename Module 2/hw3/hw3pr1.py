#
# hw3pr1.py
#
#  lab problem - matplotlib tutorial (and a bit of numpy besides...)
#
# this asks you to work through the first part of the tutorial at
#     www.labri.fr/perso/nrougier/teaching/matplotlib/
#   + then try the scatter plot, bar plot, and one other kind of "Other plot"
#     from that tutorial -- and create a distinctive variation of each
#
# include screenshots or saved graphics of your variations of those plots with the names
#   + plot_scatter.png, plot_bar.png, and plot_choice.png
#
# Remember to run  %matplotlib  at your ipython prompt!
#

#
# in-class examples...
#

import numpy as np
import matplotlib.pyplot as plt

def inclass1():
    """
    Simple demo of a scatter plot.
    """
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()



#
# First example from the tutorial/walkthrough
#


#
# Feel free to replace this code as you go -- or to comment/uncomment portions of it...
#

def example1():
    import matplotlib.cm as cm

    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C,S = np.cos(X), np.sin(X)

    plt.plot(X,C)
    plt.plot(X,S)

    plt.show()






#
# Here is a larger example with many parameters made explicit
#

def example2():
    import matplotlib.cm as cm

    # Create a new figure of size 8x6 points, using 80 dots per inch
    plt.figure(figsize=(10,6), dpi=80)

    # Create a new subplot from a grid of 1x1
    plt.subplot(111)

    X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
    C,S = np.cos(X), np.sin(X)

    # Plot cosine using blue color with a continuous line of width 1 (pixels)
    plt.plot(X, C, color="blue", linewidth=3.0, linestyle="-", label="cosine")

    # Plot sine using green color with a continuous line of width 1 (pixels)
    plt.plot(X, S, color="red", linewidth=3.0, linestyle="-", label="sine")

    # Set x limits
    plt.xlim(X.min()*1.1,X.max()*1.1)

    # Set x ticks
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            [r'$-np.pi$', r'$-np.pi*2$', r'$0$', r'$np.pi/2$', r'$np.pi$'])

    # Set y limits
    plt.ylim(C.min()*1.1,C.max()*1.1)

    # Set y ticks
    plt.yticks([-1, 0, 1],
        [r'$-1$', r'$0$', r'$1$'])

    # Set spine locations
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    # Create Legend
    plt.legend(loc='upper left', frameon=False)

    # Adjust labels for visibility
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

    # Save figure using 72 dots per inch
    # savefig("../figures/exercice_2.png",dpi=72)

    # Show result on screen
    plt.show()


def scatter_example():

    n = 1024
    x = np.random.normal(0,1,n)
    y = np.random.normal(0,1,n)
    colors = np.arctan2(y,x)

    plt.scatter(x, y, linewidth=0.25, c=colors, cmap='jet', alpha=0.65, edgecolor='black')
    # plt.xlim(-2,2)
    # plt.ylim(-2,2)
    # get rid of axes entirely
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.show()


def bar_example():

    n = 12
    X = np.arange(n)
    Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
    Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    for x,y in zip(X,Y1):
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    for x,y in zip(X,Y2):
        plt.text(x, -y-0.05, '%.2f' % y, ha='center', va= 'top')

    plt.xticks([])
    plt.yticks([])

    plt.ylim(-1.25,+1.25)
    plt.show()


def other_example():

    def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

    n = 10
    x = np.linspace(-3,3,4*n)
    y = np.linspace(-3,3,3*n)
    X,Y = np.meshgrid(x,y)
    plt.imshow(f(X,Y),cmap='bone',origin='lower')

    # get rid of axes entirely
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')

    plt.colorbar()#fraction=0.046,pad=0.04)

    plt.show()

ANIMATE = True
if ANIMATE:
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(10,10), facecolor='#40a4df')
    ax = fig.add_axes([0,0,1,1], frameon=False)#, aspect=1)

    # Number of rings
    n = 50
    size_min = 50
    size_max = 50*50
    # Ring position
    P = np.random.uniform(0,1,(n,2))
    # Ring colors
    C = np.ones((n,4)) * (0,0.0196,0.6078,1)#(0.0157,0.2431,0.7686,1)
    # Alpha color channel goes from 0 (transparent) to 1 (opaque)
    C[:,3] = np.linspace(0,1,n)
    # Ring sizes
    S = np.linspace(size_min, size_max, n)

    # Scatter plot
    scat = ax.scatter(P[:,0], P[:,1], s=S, lw=0.5, edgecolors=C, facecolors=(1,1,1,0.02))
    # Ensure limits are [0,1] and remove ticks
    ax.set_xlim(0,1), ax.set_xticks([])
    ax.set_ylim(0,1), ax.set_yticks([])
    # plt.show()

    def update(frame):
        global P,C,S

        # Every ring id made more transparent
        C[:,3] = np.maximum(0, C[:,3] - 1.0/n)
        # Each ring is made larger
        S += (size_max-size_min)/n
        # Reset ring specific ring (relative to frame number)
        i = frame % 50
        P[i] = np.random.uniform(0,1,2)
        S[i] = size_min
        C[i,3] = 1

        # Update scatter object
        scat.set_edgecolors(C)
        scat.set_sizes(S)
        scat.set_offsets(P)
        # Return modified object
        return scat,

    animation = FuncAnimation(fig, update, interval=10, blit=True, frames=200)
    plt.show()

# def main():
#     """ organize and run functions """
#
#
#
# if __name__ == "__main__":
#     main()
#
# using style sheets:
#   # be sure to               import matplotlib
#   # list of all of them:     matplotlib.style.available
#   # example of using one:    matplotlib.style.use( 'seaborn-paper' )
#
