# Importing modules 
# Choose different matplotlib back-end
#import matplotlib
#matplotlib.use('Qt5agg')
import argparse
import os, sys
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from numba import jit 
from scipy.special import gamma
import numpy as np
import scipy
import h5py

parser = argparse.ArgumentParser(description = 'Single-particle diffusion simulation')
parser.add_argument('Test', type = str, help = 'Test. (str)', default = 'Test')

parser.add_argument('-nparts', '--num-part', type = int, help = 'Number of particles to simulate (int).', default = 1) 
parser.add_argument('-dimx', '--dim-x', type = int, help = 'Dimensions in x-coordinate (int).', default = 1024)
parser.add_argument('-dimy', '--dim-y', type = int, help = 'Dimensions in y-coordinate (int).', default = 1024)
parser.add_argument('-dimz', '--dim-z', type = int, help = 'Dimensions in z-coordinate (int).', default = 1024)

parser.add_argument('-t0', '--t-start', type = int, help = 'Start of simulation (int).', default = 0)
parser.add_argument('-t1', '--t-end', type = int, help = 'End of simulation (int).', default = 1) 

parser.add_argument('-nsteps', '--num-steps', type = int, help = 'Number of time steps to simulate (int).', default = 100)
parser.add_argument('-n', '--n-time', type = int, help = 'Division of integer time (int).', default = 1)
parser.add_argument('-hurst', '--hurst-exp', type = float, help = 'Hurst exponent (float).', default = 0.5)
args = parser.parse_args()

##################### Functions 
# Reflective boundary conditions 
@jit(nopython = True, cache = True)
def square_bounds_brownian(px , py, pz, min_x, min_y, min_z, max_x, max_y, max_z, sigma):
    """ 
    Defines a square simulation boundary. 
    """
    num_p = px.shape[0]
    ti_p = px.shape[1]

    if px < min_x or px > max_x: 
        px = np.random.normal(loc = 0.0, scale = sigma)

    if py < min_y or py > max_y:
        py = np.random.normal(loc = 0.0, scale = sigma)

    if pz < min_z or pz > max_z:
        pz = np.random.normal(loc = 0.0, scale = sigma)

    return px, py, pz

@jit(nopython = True, cache = True)
def square_bounds_fbm(px , py, pz, min_x, min_y, min_z, max_x, max_y, max_z):
    """ 
    Defines a square simulation boundary. 
    """

    return px, py, pz

@jit(nopython = True, cache = True) 
def calculate_msd(pos_x, pos_y, pos_z):
    """ 
    Calculates MSD (mean squared displacement) of particle trajectories. 
    """

    return None

@jit(nopython = True, cache = True) 
def calculate_mss(pos_x, pos_y, pos_z):
    """ 
    Calculates MSS (moment scaling spectrum) of particle trajectories. 
    """

    return None

def plot_msd(t_vec , p_x, p_y, p_z): 
    """
    Plots MSD  of particle trajectories. 
    """
    print("MSD")

def plot_mss(t_vec , p_x, p_y, p_z): 
    """
    Plots MSS  of particle trajectories. 
    """
    print("MSS")

def plot_results_3d(t_vec , p_x, p_y, p_z): 
    fig = plt.figure()
    ax3d = plt.axes(projection = '3d') 
    
    for p in np.arange(0, p_x.shape[0], step = 1): 
        for t in t_vec:
            ax3d.plot3D(p_x[p, t], p_y[p, t], p_z[p, t], 'x') 
    
    ax3d.set_xlabel('X') 
    ax3d.set_ylabel('Y') 
    ax3d.set_zlabel('Z') 

def plot_results_traj(t_vec , p_x, p_y, p_z): 
    fig = plt.figure()
    ax = plt.axes() 
    
    for p in np.arange(0, p_x.shape[0], step = 1): 
        for t in t_vec: 
            plt.plot(t * 1, p_x[p, t], 'rx')
            plt.plot(t * 1, p_y[p, t], 'gx')
            plt.plot(t * 1, p_z[p, t], 'bx') 
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_ylim(200, 800)
    ax.set_title()

@jit(nopython = True, cache = True) 
def simulate_brownian(num_part, dt, t, sigma, drift = False):
    """ 
    Simulates 3D Brownian diffusion with/without drift. 
    """
    # Calculating drift components 
    if drift == True:
        v_x = np.random.random() 
        v_y = np.random.random() 
        v_z = np.random.random() 
        drift_x = v_x * dt 
        drift_y = v_y * dt 
        drift_z = v_z * dt 
    else:
        drift_x = 0 
        drift_y = 0 
        drift_z = 0 

    # Generate Brownian increments 
    increment_x = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, 1 + t.shape[0])) 
    increment_y = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, 1 + t.shape[0])) 
    increment_z = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, 1 + t.shape[0])) 

    # Pre-allocation of memory for particle positions 
    p_x = np.zeros(shape = (num_part, 1 + t.shape[0])) 
    p_y = np.zeros(shape = (num_part, 1 + t.shape[0])) 
    p_z = np.zeros(shape = (num_part, 1 + t.shape[0])) 

    # Generate initial position of particle(s) 
    p_x[:, 0] = x0 + 1 * np.random.random(size = (1, num_part)) 
    p_y[:, 0] = y0 + 1 * np.random.random(size = (1, num_part)) 
    p_z[:, 0] = z0 + 1 * np.random.random(size = (1, num_part)) 

    for p in np.arange(0, num_part, step = 1): 
        for ti in t: 
            p_x[p, ti] = p_x[p, ti - 1] + increment_x[p, ti] + 10 * drift_x 
            p_y[p, ti] = p_y[p, ti - 1] + increment_y[p, ti] + 10 * drift_y 
            p_z[p, ti] = p_z[p, ti - 1] + increment_z[p, ti] + 10 * drift_z 
    
    return p_x, p_y, p_z

@jit(nopython = True, cache = True) 
def simulate_fractionalbrownian(num_part, H, M, n, t, dt, x0, y0, z0, gamma_H): 
    """ 
    Simulates 3D fractional Brownian motion. 
    """
    # Generate zero mean and unit variance increments 
    incx = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, n * (M + t.shape[0]))) 
    incy = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, n * (M + t.shape[0]))) 
    incz = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, n * (M + t.shape[0]))) 

    # Pre-allocation of memory for particle positions 
    p_x = np.zeros(shape = (num_part, n * (M + t.shape[0]))) 
    p_y = np.zeros(shape = (num_part, n * (M + t.shape[0]))) 
    p_z = np.zeros(shape = (num_part, n * (M + t.shape[0]))) 

    # Generate initial position of particle(s)
    p_x[:, 0] = x0 + 1 * np.random.random(size = (1, num_part)) 
    p_y[:, 0] = y0 + 1 * np.random.random(size = (1, num_part)) 
    p_z[:, 0] = z0 + 1 * np.random.random(size = (1, num_part)) 

    const = (n ** - H)/ gamma_H 
    check_x = np.zeros(shape = (num_part, n * (M + t.shape[0])))
    check_y = np.zeros(shape = (num_part, n * (M + t.shape[0])))
    check_z = np.zeros(shape = (num_part, n * (M + t.shape[0])))
    
    for p in np.arange(0, num_part, step = 1): 
        for ti in t:

            s1_x = np.array([ ((i ** (H - 0.5)) * incx[p, 1 + n * (0*M + ti) - i]) for i in range(1, n + 1)]).sum() 
            s2_x = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incx[p, 1 + n * (M - 1 + ti) - i]) for i in range(1, 1 + n * (M - 1))]).sum() 
            s1_y= np.array([ ((i ** (H - 0.5)) * incy[p, 1 + n * (0*M + ti) - i]) for i in range(1, n + 1)]).sum() 
            s2_y = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incy[p, 1 + n * (M - 1 + ti) - i]) for i in range(1, 1 + n * (M - 1))]).sum() 
            s1_z = np.array([ ((i ** (H - 0.5)) * incz[p, 1 + n * (0*M + ti) - i]) for i in range(1, n + 1)]).sum() 
            s2_z = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incz[p, 1 + n * (M - 1 + ti) - i]) for i in range(1, 1 + n * (M - 1))]).sum() 

            p_x[p, ti] = p_x[p, ti - 1] + const * (s1_x + s2_x) 
            p_y[p, ti] = p_y[p, ti - 1] + const * (s1_y + s2_y) 
            p_z[p, ti] = p_z[p, ti - 1] + const * (s1_z + s2_z) 

    return p_x, p_y, p_z

# Parameters of the simulation 
num_part = args.num_part
x_dim = args.dim_x 
y_dim = args.dim_y
z_dim = args.dim_z 

t_0 = args.t_start
t_end = args.t_end
n_steps = args.num_steps + 1
dt = (t_end - t_0) / n_steps

H = args.hurst_exp
n = args.n_time

M = 300 
D = 1 
sigma = np.sqrt(2 * dt * D)
gamma_H = gamma(H + 0.5)

# Define origin of simulation
origin = x_dim / 2 
x0 = origin 
y0 = origin 
z0 = origin 

# If we choose n = 1 we do normal Brownian diffusion 
if n == 1:
    t = np.arange(start = t_0 + 1, stop = n_steps, step = 1) 
    p_x, p_y, p_z = simulate_brownian(num_part, dt, t, sigma, drift = True) 

    plot_results_3d(t, p_x, p_y, p_z) 
    plot_results_traj(t, p_x, p_y, p_z) 
else: 
    t_vec = np.arange(start = t_0 + 1, stop = n * (n_steps + M), step = 1) 
    p_x_frac, p_y_frac, p_z_frac = simulate_fractionalbrownian(num_part, H, M, n, t_vec, dt, x0, y0, z0, gamma_H)

    plot_results_3d(t_vec, p_x_frac, p_y_frac, p_z_frac)
    plot_results_traj(t_vec, p_x_frac, p_y_frac, p_z_frac)

plt.show() 