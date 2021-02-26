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

parser = argparse.ArgumentParser(description = 'Single-particle diffusion simulation')
parser.add_argument('Test', type = str, help = 'Test. (str)', default = 'Test')

parser.add_argument('-nparts', '--num-part', type = int, help = 'Number of particles to simulate (int).', default = 1) 
parser.add_argument('-nsteps', '--num-steps', type = int, help = 'Number of time steps to simulate (int).', default = 100)
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
def initial_position(pos_x, pos_y, pos_z, x_0, y_0, z_0, x_1, y_1, z_1):
    """ 
    Generates initial positions for particle(s). 
    """

    return p_x, p_y, p_z

@jit(nopython = True, cache = True)
def calculate_msd(pos_x, pos_y, pos_z):
    """ 
    Calculates MSD of particle trajectories. 
    """

    return None

def plot_msd(t_vec , p_x, p_y, p_z): 
    """
    Plots MSD of particle trajectories. 
    """
    print("MSD")

def plot_results_3d(t_vec , p_x, p_y, p_z): 
    fig = plt.figure()
    ax3d = plt.axes(projection = '3d') 
    
    for p in np.arange(0, p_x.shape[0], step = 1): 
        for t in t_vec:
            ax3d.plot3D(p_x[p, t], p_y[p, t], p_z[p, t], 'o') 
    
    ax3d.set_xlabel('X') 
    ax3d.set_ylabel('Y') 
    ax3d.set_zlabel('Z') 

def plot_results_traj(t_vec , p_x, p_y, p_z): 
    fig = plt.figure()
    
    for p in np.arange(0, p_x.shape[0], step = 1): 
        for t in t_vec:
            plt.plot(t * dt, p_x[p, t], 'rx')
            plt.plot(t * dt, p_y[p, t], 'go')
            plt.plot(t * dt, p_z[p, t], 'b*') 

@jit(nopython = True, cache = True) 
def simulate_brownian(num_part, p_x, p_y, p_z, increment_x, increment_y, increment_z, dt, t, drift = False, sigma):
    """ 
    Simulates 3D Brownian diffusion with/without drift. 
    """
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

    # Generate initial conditions

    for p in np.arange(0, num_part, step = 1): 
        for ti in t: 
            p_x[p, ti] = p_x[p, ti - 1] + increment_x[p, ti] + drift_x 
            p_y[p, ti] = p_y[p, ti - 1] + increment_y[p, ti] + drift_y 
            p_z[p, ti] = p_z[p, ti - 1] + increment_z[p, ti] + drift_z 
            
            #p_x[p, ti], p_y[p, ti], p_z[p, ti] = square_bounds_brownian(p_x[p, ti], p_y[p, ti], p_z[p, ti], 0, 0, 0, x_dim, y_dim, z_dim, sigma) 
    
    return p_x, p_y, p_z

@jit(nopython = True, cache = True) 
def simulate_fractionalbrownian(num_part, H, M, n, t, dt, x0, y0, z0, gamma_H): 
    """ 
    Simulates 3D fractional Brownian motion. 
    """

    incx = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, n * (0*M + t.shape[0]))) 
    incy = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, n * (0*M + t.shape[0]))) 
    incz = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, n * (0*M + t.shape[0]))) 

    p_x = np.zeros(shape = (num_part, n * (0*M + t.shape[0]))) 
    p_y = np.zeros(shape = (num_part, n * (0*M + t.shape[0]))) 
    p_z = np.zeros(shape = (num_part, n * (0*M + t.shape[0]))) 

    # Generate initial conditions
    p_x[:, 0] = x0 + 1 * np.random.random(size = (1, num_part)) 
    p_y[:, 0] = y0 + 1 * np.random.random(size = (1, num_part)) 
    p_z[:, 0] = z0 + 1 * np.random.random(size = (1, num_part)) 

    const = (n ** - H)/ gamma_H 
    check_x = np.zeros(shape = (num_part, n * (0*M + t.shape[0])))
    check_y = np.zeros(shape = (num_part, n * (0*M + t.shape[0])))
    check_z = np.zeros(shape = (num_part, n * (0*M + t.shape[0])))
    
    for p in np.arange(0, num_part, step = 1): 
        for ti in t:

            s1_x = np.array([ ((i ** (H - 0.5)) * incx[p, 1 + n * (0*M + ti) - i]) for i in range(1, n + 0)]).sum() 
            s2_x = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incx[p, 1 + n * (0*M - 1 + ti) - i]) for i in range(1, 0 + n * (M - 1))]).sum() 
            s1_y= np.array([ ((i ** (H - 0.5)) * incy[p, 1 + n * (0*M + ti) - i]) for i in range(1, n + 0)]).sum() 
            s2_y = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incy[p, 1 + n * (0*M - 1 + ti) - i]) for i in range(1, 0 + n * (M - 1))]).sum() 
            s1_z = np.array([ ((i ** (H - 0.5)) * incz[p, 1 + n * (0*M + ti) - i]) for i in range(1, n + 0)]).sum() 
            s2_z = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incz[p, 1 + n * (0*M - 1 + ti) - i]) for i in range(1, 0 + n * (M - 1))]).sum() 

            p_x[p, ti] = p_x[p, ti - 1] + const * (s1_x + s2_x) 
            p_y[p, ti] = p_y[p, ti - 1] + const * (s1_y + s2_y) 
            p_z[p, ti] = p_z[p, ti - 1] + const * (s1_z + s2_z) 
            check_x[p, ti] = const * (s1_x + s2_x) 
            check_y[p, ti] = const * (s1_y + s2_y) 
            check_z[p, ti] = const * (s1_z + s2_z) 
            
            # Fix: pass in current position for current particle p at current time step ti and check whether boundary condition(s) are met.... 
            #p_x[p, ti], p_y[p, ti], p_z[p, ti] = square_bounds(p_x[p, ti], p_y[p, ti], p_z[p, ti], 0, 0, 0, x_dim, y_dim, z_dim) 

    return p_x, p_y, p_z, check_x, check_y, check_z

#![test](./images/Figure_2_0.1.png)

# Simulation parameters 
num_part = args.num_part
x_dim = 1024 
y_dim = x_dim
z_dim = x_dim 

t_0 = 0
t_end = 1
n_steps = args.num_steps 
dt = (t_end - t_0) / n_steps

H = args.hurst_exp
gamma_H = gamma(H + 0.5)
n = 3
M = 300

D = 1
sigma = np.sqrt(2 * dt * D)

# Generate origin coordinates
origin = x_dim / 2
x0 = origin
y0 = origin
z0 = origin 

# Generate Brownian increments - put inside function
increment_x = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, n_steps)) 
increment_y = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, n_steps)) 
increment_z = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, n_steps)) 

# Pre-allocation of memory for particle positions - put inside function
p_x = np.zeros(shape = (num_part, n_steps)) 
p_y = np.zeros(shape = (num_part, n_steps)) 
p_z = np.zeros(shape = (num_part, n_steps)) 

# Initial position of particle - put inside function
p_x[:, 0] = x0 + 5 * np.random.random(size = (1, num_part)) 
p_y[:, 0] = y0 + 5 * np.random.random(size = (1, num_part)) 
p_z[:, 0] = z0 + 5 * np.random.random(size = (1, num_part)) 

# Create time vector for simulation 
t_vec = np.arange(start = t_0 + 1, stop = 0 + n * (n_steps + M), step = 1) # 2800 size
# Call simulation functions 

#p_x, p_y, p_z = simulate_brownian(num_part, p_x, p_y, p_z, increment_x, increment_y, increment_z, dt, t_vec, drift = False, sigma) 

p_x_frac, p_y_frac, p_z_frac, check_x, check_y, check_z = simulate_fractionalbrownian(num_part, H, M, n, t_vec, dt, x0, y0, z0, gamma_H)

fig = plt.figure()
for p in np.arange(0, check_x.shape[0], step = 1): 
    for t in t_vec: 
        plt.plot(dt*(t), check_x[p, t], 'ro')
        plt.plot(dt*(t), check_y[p, t], 'go')
        plt.plot(dt*(t), check_z[p, t], 'bo') 
#plt.show()
plot_results_3d(t_vec, p_x_frac, p_y_frac, p_z_frac)
plot_results_traj(t_vec, p_x_frac, p_y_frac, p_z_frac)
plt.show()

plotting = 0
if plotting == 1: 
    plot_results_3d(t_vec , p_x, p_y, p_z)
    plot_results_traj(t_vec , p_x, p_y, p_z)
    plt.show()

