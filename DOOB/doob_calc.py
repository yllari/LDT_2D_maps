import numpy as np
from numba import njit

# ---------------- Doob transform ----------------


def gamma(x_center, x_lin, y_lin, div,
          cum_doob, cum_invar):
    ''' Construction of gamma function,
    :param x_center: 2x(NxN) array of x and y coordinates
    :param x_lin: 1d array-like of bins for x coordinates
    :param y_lin: 1d array-like of bins for y coordinates
    :param cum_doob: [F_X, F_{Y|X}] for doob map
    :param cum_invar: [F_X,F_{Y|X}] for base map

    :return: 2x(NxN) array with [(2d)x-coords, (2d)y-coords]
    '''
    # This digitize may not be necessary if the function is redone
    dx = (x_lin[1]-x_lin[0])
    dy = (y_lin[1]-y_lin[0])

    x_cd = np.digitize(x_center[0],bins=x_lin,right=False) - 1
    y_cd = np.digitize(x_center[1],bins=y_lin,right=False) - 1

    # Inverse coordinates
    x_invcoord = np.searchsorted(cum_doob[0], v = cum_invar[1][x_cd]) - 1

    # This is a hotfix because in cum_invar, the zero value to use as bins is
    # not included because of the cumsum, but should be
    x_invcoord[x_invcoord == -1] = 0
    y_invcoord = classify(x_invcoord, x_cd, y_cd, cum_doob[1], cum_invar[1])
    y_invcoord[y_invcoord == -1] = 0

    coords = np.array([x_invcoord,y_invcoord])

    # Transforming to position coordinates again
    coords[0] = x_lin[0] + coords[0]*dx + dx/2.
    coords[1] = y_lin[0] + coords[1]*dy + dy/2.
    return coords


def gamma_inv(x_center, x_lin, y_lin, div,
              cum_doob, cum_invar):
    ''' Construction of gamma inverse function,
    :param x_center: 2x(NxN) array of x and y coordinates
    :param x_lin: 1d array-like of bins for x coordinates
    :param y_lin: 1d array-like of bins for y coordinates
    :param cum_doob: [F_X,F_{Y|X}] for doob invariant density
    :param cum_invar: [F_X,F_{Y|X}] for base invariant density

    :return: 2x(NxN) array with [(2d)x-coords, (2d)y-coords]
    '''
    dx = (x_lin[1]-x_lin[0])
    dy = (y_lin[1]-y_lin[0])
    x_cd = np.digitize(x_center[0], bins=x_lin, right=False)-1
    y_cd = np.digitize(x_center[1], bins=y_lin, right=False)-1

    x_invcoord = np.searchsorted(cum_invar[0], v = cum_doob[0][x_cd]) - 1
    x_invcoord[x_invcoord == -1] = 0
    y_invcoord = classify(x_invcoord, x_cd, y_cd, cum_invar[1], cum_doob[1])
    y_invcoord[y_invcoord == -1] = 0

    # Transforming to position coordinates again
    coords = np.array([x_invcoord, y_invcoord])
    coords[0] = x_lin[0] + coords[0]*dx + dx/2.
    coords[1] = y_lin[0] + coords[1]*dy + dy/2.

    return coords

@njit(parallel = True)
def classify(
        x_gam:np.array,
        x:np.array,
        y:np.array,
        cum_1:np.array,
        cum_2:np.array):
    ''' A function that computes the 'y' coordinate as the inverse
    '''

    y_invcoord = np.zeros(x.shape)
    cum_1 = cum_1.T
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = cum_1[x_gam[i,j]]
            b = cum_2[y[i,j],x[i,j]]
            y_invcoord[i,j] = np.searchsorted(a,b) - 1
    return y_invcoord

# Calculating doob map

def doob(invariant, l, r,
         f,
         x_center, map_range, div
         ):
    ''' Calculate doob map following same scheme as in all code
    param: invariant: 2d array of invariant density of base map
    param: l:  left eigenvector
    param: r:  right eigenvector
    param: f:  map
    param: x_center: 2x(NxN) center of coordinates
    param: map_range: self explanatory [x_min,x_max,y_min,y_max]
    param: div: divisions, div=N

    return: gamma function, doob function
    '''
    ## Coordinates calculations
    x_min, x_max, y_min, y_max = map_range
    x_lin = np.linspace(x_min,x_max,div+1)
    y_lin = np.linspace(y_min,y_max,div+1)
    dx = (x_max-x_min)/div
    dy = (y_max-y_min)/div

    # The function [CDF[marginal_x],F_{Y|X}
    # (CDF of conditioned probability)]
    P_x = np.sum(r*l,axis=0)*dy
    cum_doob = [np.cumsum(P_x)*dx,
                np.cumsum(r*l,axis=0)/P_x*dy] # Row-Wise division

    P_x = np.sum(invariant,axis = 0)*dy
    cum_invar = [np.cumsum(P_x)*dx,
                 np.cumsum(invariant,axis=0)/P_x*dy]
    #
    #Calculating gamma(f(gamma^{-1}))
    gam_1 = gamma_inv(x_center, x_lin, y_lin, div, cum_doob, cum_invar)
    f_gam = f(gam_1)
    doob_map = gamma(f_gam, x_lin, y_lin, div, cum_doob, cum_invar)
    return gam_1, doob_map
