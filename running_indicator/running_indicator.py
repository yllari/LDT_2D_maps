import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor
import argparse
import time

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


def g(x, c, c_size):
    '''Returns the observable value on a given point

    :paran x: point to evaluate
    :param c: index for center of indicator funciton
    :param c_size:  index size for indicator
    output:
    '''

    x_c = x[0,:]
    y_c = x[1,:]
    a = np.zeros(x_c.shape)
    a[c[1] - c_size:c[1] + c_size, c[0] - c_size:c[0] + c_size] = 1

    return a
def r_iter(
        map_range, div, iters,
        alpha,
        f_inv, f_prime, g, g_params
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
    c_center, c_size = g_params
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
    z_indx0 = (z[0,:]/dx).astype(int)
    z_indx1 = (z[1,:]/dy).astype(int)


    g_x = g(x_center, c_center, c_size)
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

def running_indic(params, c_list, c_size, ncores):
    ''' Parallelized running indic calculation'''
    values = []
    map_range, div, iters, alpha, f_inv, f_prime, g = params

    # Parallelization

    sort_res = []
    procs = []
    with ProcessPoolExecutor(ncores) as executor:
        for c in c_list:
            proc = executor.submit(r_iter, map_range, div, iters, alpha, f_inv, f_prime, g, [c, c_size])
            procs.append(proc)
        for i in range(len(c_list)):
            sort_res.append([c_list[i], procs[i].result()])

    return sort_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ParallelIndic',
                    description='Parallelized calculation of running indicator function with right eigenfunction',
                    epilog='YGK 2024')
    parser.add_argument('div', type=int)
    parser.add_argument('ncores', type=int)
    parser.add_argument('c_size', type=int)

    div = parser.parse_args().div
    ncores = parser.parse_args().ncores
    c_size = parser.parse_args().c_size



    #-+-+-+-+-+-+-+-+-+-+-+- PARAMETERS +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # map_range -- [x_min,x_max,y_min,y_,max]
    x_min = 0
    x_max = 1.
    y_min = 0.
    y_max = 1.
    map_range = [x_min,x_max,y_min,y_max]
    # iters -- number of iterations for power method
    iters = 5

    # div -- discretization of space (same for both directions), div x div
    alpha = 3
    dx = (x_max - x_min)/div
    dy = (y_max - y_min)/div

    c_size = int(div/c_size)
    n_discret = c_size# number of indicator functions in each axis
    c_list = np.arange(c_size, div, n_discret)
    c_list[-1] -= 1
    print(len(c_list))
    c_listx = np.meshgrid(c_list,c_list)[0].flatten()
    c_listy = np.meshgrid(c_list,c_list)[1].flatten()
    c_list = [ [c_listx[i], c_listy[i]] for i in range(len(c_listx))]
    # Rearranging

    params = map_range, div, iters, alpha, f_inv, f_prime, g
    t_0 = time.time()
    res = running_indic(params, c_list, c_size, ncores)
    print("Elapsed time:", time.time() - t_0)

    coords = np.array([res[i][0] for i in range(len(res))]).T/div
    value = [res[i][1] for i in range(len(res))]
    np.savetxt("coords.txt", coords)
    np.savetxt("value.txt", value)
    plt.scatter(coords[1], coords[0], c=value)
    plt.colorbar()
    plt.show()
