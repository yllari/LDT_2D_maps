import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
rc_fonts = {
    "font.family": "STIXGEneral",
    "font.size": 18,
    'figure.figsize': (7, 5),
}
mpl.rcParams.update(rc_fonts)
mpl.rcParams['mathtext.fontset'] = 'stix'
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_1d(x,y,name,labels):
    ''' Plotting in 1d pre-fabricated function
        :param x: array-like of x-coordinates
        :param y: array-like of y-coordinates
        :param name: title of plot
        :param labels: array-like with x_label in first coordinate,
               y_label in second
        :return: saved figure as file "name.png"
    '''
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(name,dpi=300)

def plot_2d(h,name,labels,extents):
    ''' Plotting in 1d pre-fabricated function
        :param h: 2d-array-like of values
        :param name: title of plot
        :param labels: array-like with x_label in first coordinate,
               y_label in second and legend for colorbar in third
        :param extents: limits of plot
        :return: saved figure as file "name.png"
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(h,origin="lower",extent=extents)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im,cax,orientation="vertical")
    cb.set_label(labels[2])
    fig.tight_layout()
    fig.savefig(name,dpi=300)

def plot_2dlog(h,name,labels,extents):
    fig, ax = plt.subplots()
    im = ax.imshow(h,origin="lower",norm=colors.LogNorm(),extent=extents, cmap="inferno")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(name)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im,cax,orientation="vertical")
    cb.set_label(labels[2])
    fig.tight_layout()
    fig.savefig(name,dpi=300)


def plot_2dpow(h,name,labels,extents):
    fig, ax = plt.subplots()
    im = ax.imshow(h,origin="lower",norm=colors.PowerNorm(0.5),extent=extents, cmap="inferno")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im,cax,orientation="vertical")
    cb.set_label(labels[2])
    fig.tight_layout()
    fig.savefig(name,dpi=300)
