import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from doob_calc import doob
from func_plot import plot_1d
from func_plot import plot_2d
from func_plot import plot_2dlog
from func_plot import plot_2dpow

#-+-+-+-+-+-+-+-+-+-+-+- PARAMETERS +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
# map_range -- [x_min,x_max,y_min,y_,max]
x_min = 0
x_max = 1.
y_min = 0.
y_max = 1.
map_range = [x_min,x_max,y_min,y_max]
# iters -- number of iterations for power method
iters = 10
# div -- discretization of space (same for both directions), div x div
div = int(1e3)
dx = (x_max - x_min)/div
dy = (y_max - y_min)/div
# alpha -- alpha parameter for tilting
alpha = 1

# save_maps -- whether to save the doob maps in a txt file
save_maps = False
# save_rho_D -- whether to save the new invariant density rho_D
save_rho_D = False


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

    print(np.sum(z_indx1==1000))

    g_x = g(x_center)
    g_z = g_x[z_indx1, z_indx0]
    fprime_x = f_prime(x_center)
    fprime_z = fprime_x[z_indx1, z_indx0]

    r_ev_list = [1]
    print(f"######Finding rights with alpha={alpha} ########")
    for i in range(0,iters):
        # This can be optimized, expressed like this for clarity
        r_aux = np.exp(alpha*g_z)*r[z_indx1, z_indx0]/fprime_z
        # r = np.sum(r_aux,axis=0) #Uncomment if f is not bijective
        r_ev = np.sum(r_aux)*dx*dy
        print(i, r"lambda="+str(r_ev))
        # Renormalization
        r = r_aux/r_ev
        r_ev_list.append(r_ev)
    return r, r_ev_list


def l_iter(
        map_range, div, iters,
        alpha,
        r, f, g):

    ''' Return the right eigenfunction using the power method iteration
    :param map_range: list with [x_min, x_max, y_min, y_max]
    :param div: number of divisions on both x and y axis
    :param iters: number of iterations for the power method
    :param alpha: control parameter of the tilting
    :param r: right eigenfunction as calulated by 'r_iter'
    :param f: map function
    :param g: function observable depending of positions
    :return: right eigenfunction as a NxN matrix, right eigenvalue
    '''
    a, b, c, d = map_range
    # Discrete grid of two-dimensional map, "div" equally spaced bins
    dx = (b-a)/div
    dy = (d-c)/div

    x_lin = np.linspace(a, b, div+1)
    y_lin = np.linspace(c, d, div+1)

    x = np.meshgrid(x_lin[1:], y_lin[1:])
    x = np.array([x[0], x[1]])
    x_center = x
    x_center[0] -= dx/2.
    x_center[1] -= dy/2.

    # Starting values
    l = np.ones((div,div))
    l_ev = 0  # Safety measure, will blow up if bad-initialized

    g_x = g(x_center)

    # Same thing, redundant but maybe can make it clearer
    f_eval = f(x_center)
    x_indx = (f_eval[0,:]/dx).astype(int)
    y_indx = (f_eval[1,:]/dy).astype(int)
    l_ev_list = [1]
    print(f"######Finding lefts with alpha={alpha} ########")
    for i in range(0, iters):
        l_aux = np.exp(alpha*g_x)*l[y_indx, x_indx]
        l_ev = np.sum(l_aux*r)*dx*dy
        print(i, "lambda="+str(l_ev))
        l = l_aux/l_ev
        l_ev_list.append(l_ev)
    return l, l_ev_list


if __name__ == "__main__":
    # -+-++-+-+-+ Power method, calculating invariant, right and left --+-+*
    invariant, lambda_inv = r_iter(map_range, div, iters, 0, f_inv, f_prime, g)
    r, r_ev = r_iter(map_range, div, iters, alpha, f_inv, f_prime, g)
    l, l_ev = l_iter(map_range, div, iters, alpha, r, f, g)
    r_ev = [np.log(x) for x in r_ev]
    l_ev = [np.log(x) for x in l_ev]
    n_list = np.arange(0, iters + 1, 1)
    print(n_list)
    print(len(r_ev))
    fig, ax = plt.subplots()
    ax.scatter(n_list, r_ev, edgecolor="red", facecolor="none" ,label="From right")
    ax.scatter(n_list, l_ev, edgecolor="blue", facecolor="none", label="From left")
    inset = ax.inset_axes([0.4,0.5, 0.5, 0.3],)
    inset.scatter(n_list[1:], r_ev[1:], edgecolor="red", facecolor="none")
    inset.scatter(n_list[1:], l_ev[1:], edgecolor="blue", facecolor="none")
    inset.set_xlabel(r"$i$")
    ax.indicate_inset_zoom(inset, edgecolor="black")
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\theta$(s=1)")
    ax.set_title(f"N={div}")
    ax.legend(loc="lower center")
    fig.savefig("Convergence.png", dpi=300)
