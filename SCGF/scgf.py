import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor
import time
import argparse

def f(x):
    ''' Insert here the map

    :param x: 2x(NxN) [(2d)x,(2d)y]
        (array) with representing coordiantes
    :return: 2x(NxN) (array), Discrete map value
    '''
    x_c = x[0,:]
    y_c = x[1,:]

    result = np.array([
        np.fmod( x_c +   y_c, 1.),
        np.fmod( x_c + 2*y_c, 1.)]
    )

    return result


def f_inv(x):
    ''' Returns the inverse function of a given value, can be n-sized array
    :param x: given value
    :return: inverse function value
    '''
    x_c = x[0,:]
    y_c = x[1,:]

    result  = np.array([
        np.fmod( 2*x_c - y_c + 1, 1.),
        np.fmod(   y_c - x_c + 1, 1.) ]
    )

    return result


def f_prime(x):
    ''' Returns the Jacobian evaluated on a given point
    :param x: point to evaluate
    :return: determinant of Jacobian evaluated at point
    '''

    return (np.ones(x.shape)*1.)[0,:]


def g(x):
    ''' Returns the observable value on a given point
    :param x: point to evaluate
    :return: Observable associated
    '''

    x_c = x[0,:]
    y_c = x[1,:]

    return 0.5*(x_c+y_c)


def r_iter(
        map_range, div, iters,
        alpha,
        f_inv, f_prime, g
    ):
    ''' Return the right eigenfunction using the power method iteration
    :param map_range: list with [x_min, x_max, y_min, y_max]
    :param div: number of divisions on both x and y axis
    :param iters: number of iterations for the power method
    :param alpha: control parameter of the tilting
    :param f_inv: inverse map as a defined function
    :param f_prime: function that returns the determinatn of the jacobian at each point
    :param g: function observable depending of positions
    :return: right eigenfunction as a NxN matrix, right eigenvalue
    '''

    a, b, c, d = map_range
    # Discrete grid of two-dimensional map, "div" equally spaced bins
    dx = (b-a)/div
    dy = (d-c)/div
    x_lin = np.linspace(a, b, div+1)
    y_lin = np.linspace(c, d, div+1)

    # Starting values
    r = np.ones((div, div))
    r_ev = 0 # Safety measure, will blow up if bad-initialized

    # Given our discrete map, the inverse values will not change. The
    # same can be said about our observable g(z). They both can be calculated beforehand
    x = np.meshgrid(x_lin[1:], y_lin[1:])
    x = np.array([x[0], x[1]])
    x_center = x

    # Actually center the values to evaluate f(x,y)
    x_center[0] -= dx/2.
    x_center[1] -= dy/2.
    z = f_inv(x_center)

    # Find index associated to each inverse, x_{i-1} =< z <x_i
    z_indx0 = (z[0,:]/dy).astype(int)
    z_indx1 = (z[1,:]/dx).astype(int)


    g_x = g(x_center)
    g_z = g_x[z_indx1, z_indx0]
    fprime_x = f_prime(x_center)
    fprime_z = fprime_x[z_indx1, z_indx0]
    for i in range(0,iters):
        # This can be optimized, expressed like this for clarity
        r_aux = np.exp(alpha*g_z)*r[z_indx1, z_indx0]/fprime_z
        # r = np.sum(r_aux,axis=0) #Uncomment if f is not bijective
        r_ev = np.sum(r_aux)*dx*dy
        # Renormalization
        r = r_aux/r_ev
    return np.log(r_ev)

def SCGF(params, alpha_range,steps,iters, ncores):
    ''' Parallelized SCGF calculation'''
    steps = np.linspace(alpha_range[0],alpha_range[1],steps)
    th_r = []
    map_range, div, f_inv, f_prime, g = params

    # Parallelization

    data = []
    procs = []
    with ProcessPoolExecutor(ncores) as executor:
        for alpha_ in steps:
            proc = executor.submit(r_iter, map_range, div, iters, alpha_, f_inv, f_prime, g)
            procs.append(proc)
        for i in range(len(steps)):
            data.append([steps[i], procs[i].result()])
    return np.array(data).T

if __name__ == "__main__":
    #-+-+-+-+-+-+-+-+-+-+-+- PARAMETERS +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # map_range -- [x_min,x_max,y_min,y_,max]
    x_min = 0
    x_max = 1.
    y_min = 0.
    y_max = 1.
    map_range = [x_min,x_max,y_min,y_max]
    # iters -- number of iterations for power method

    parser = argparse.ArgumentParser(
                    prog='ParallelSCGF',
                    description='Parallelized calculation of SCGF with right eigenfunction',
                    epilog='YGK 2024')
    parser.add_argument('div', type=int, help='Discretization of space, number of divisions in each axis')
    parser.add_argument('iters', type=int, help='Number of iterations per s=alpha=k')
    parser.add_argument('ncores', type=int, help='Number of cores to use')

    div = parser.parse_args().div
    iters = parser.parse_args().iters
    ncores = parser.parse_args().ncores

    # div -- discretization of space (same for both directions), div x div
    alpha_range = [-7,7]
    steps = 60

    dx = (x_max - x_min)/div
    dy = (y_max - y_min)/div
    params = map_range, div, f_inv, f_prime, g
    t_0 = time.time()
    st_, th_r = SCGF(params, alpha_range, steps, iters, ncores)
    print("Elapsed time:", time.time() - t_0)
    np.savetxt("SCGF.dat", np.array([st_, th_r]).T)
