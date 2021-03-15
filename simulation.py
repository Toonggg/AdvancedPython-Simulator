# Importing modules 
import argparse
import numpy as np
import scipy
import h5py 
import os, sys
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from scipy.special import gamma
from numba import jit
import tkinter

# Parser arguments
parser = argparse.ArgumentParser(description = 'Single-particle diffusion simulation') 
parser.add_argument('Test', type = str, help = 'Test. (str)', default = 'Test') # what should be put here??? 

parser.add_argument('-nparts', '--num-part', type = int, help = 'Number of particles to simulate (int).', default = 1) 
parser.add_argument('-dimx', '--dim-x', type = int, help = 'Dimensions in x-coordinate (int).', default = 1024) 
parser.add_argument('-dimy', '--dim-y', type = int, help = 'Dimensions in y-coordinate (int).', default = 1024) 
parser.add_argument('-dimz', '--dim-z', type = int, help = 'Dimensions in z-coordinate (int).', default = 1024) 

parser.add_argument('-t0', '--t-start', type = int, help = 'Start of simulation (int).', default = 0) 
parser.add_argument('-t1', '--t-end', type = int, help = 'End of simulation (int).', default = 1) 

parser.add_argument('-nsteps', '--num-steps', type = int, help = 'Number of time steps to simulate (int).', default = 100) 
parser.add_argument('-M', '--num-M', type = int, help = 'Range covered in time (int).', default = 300) 
parser.add_argument('-n', '--n-time', type = int, help = 'Division of integer time (int).', default = 1) 
parser.add_argument('-hurst', '--hurst-exp', type = float, help = 'Hurst exponent (float).', default = 0.5) 
args = parser.parse_args()

##################### Functions BEGINNING #####################
@jit(nopython = True, cache = True) 
def simulate_brownian(num_part, dt, time_steps, x0, y0, z0, sigma, drift = False): 
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
    increment_x = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, time_steps)) 
    increment_y = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, time_steps)) 
    increment_z = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, time_steps)) 

    # Pre-allocation of memory for particle positions 
    p_x = np.zeros(shape = (num_part, time_steps)) 
    p_y = np.zeros(shape = (num_part, time_steps))
    p_z = np.zeros(shape = (num_part, time_steps))

    # Generate initial position of particle(s) 
    p_x[:, 0] = x0 + 20 * np.random.random(size = (1, num_part)) 
    p_y[:, 0] = y0 + 20 * np.random.random(size = (1, num_part)) 
    p_z[:, 0] = z0 + 20 * np.random.random(size = (1, num_part)) 

    for p in np.arange(0, num_part, step = 1): 
        for ti in np.arange(start = 1, stop = time_steps, step = 1): 
            p_x[p, ti] = p_x[p, ti - 1] + increment_x[p, ti] + 10 * drift_x 
            p_y[p, ti] = p_y[p, ti - 1] + increment_y[p, ti] + 10 * drift_y 
            p_z[p, ti] = p_z[p, ti - 1] + increment_z[p, ti] + 10 * drift_z 

    return p_x, p_y, p_z

@jit(nopython = True, cache = True) 
def simulate_fractionalbrownian(num_part, H, M, n, t, x0, y0, z0, gamma_H): 
    """ 
    Simulates 3D fractional Brownian motion.    
    """
    # Generate zero mean and unit variance increments 
    incx = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, t.shape[0])) 
    incy = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, t.shape[0])) 
    incz = np.random.normal(loc = 0.0, scale = 1.0, size = (num_part, t.shape[0])) 

    # Pre-allocation of memory for particle positions 
    p_x = np.zeros(shape = (num_part, t.shape[0])) 
    p_y = np.zeros(shape = (num_part, t.shape[0])) 
    p_z = np.zeros(shape = (num_part, t.shape[0])) 

    # Generate initial position of particle(s)
    p_x[:, 0] = x0 + 20 * np.random.random(size = (1, num_part)) 
    p_y[:, 0] = y0 + 20 * np.random.random(size = (1, num_part)) 
    p_z[:, 0] = z0 + 20 * np.random.random(size = (1, num_part)) 

    const = (n ** - H)/ gamma_H 
    
    for p in np.arange(0, num_part, step = 1): 
        for ti in t: 

            s1_x = np.array([ ((i ** (H - 0.5)) * incx[p, 1 + ti - i]) for i in range(1, n + 1)]).sum() 
            s2_x = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incx[p, 1 + ti - n - i]) for i in range(1, 1 + n * (M - 1))]).sum() 
            s1_y = np.array([ ((i ** (H - 0.5)) * incy[p, 1 + ti - i]) for i in range(1, n + 1)]).sum() 
            s2_y = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incy[p, 1 + ti - n - i]) for i in range(1, 1 + n * (M - 1))]).sum() 
            s1_z = np.array([ ((i ** (H - 0.5)) * incz[p, 1 + ti - i]) for i in range(1, n + 1)]).sum() 
            s2_z = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incz[p, 1 + ti - n - i]) for i in range(1, 1 + n * (M - 1))]).sum() 

            icx = const * (s1_x + s2_x) 
            icy = const * (s1_y + s2_y) 
            icz = const * (s1_z + s2_z) 

            p_x[p, ti] = p_x[p, ti - 1] + icx 
            p_y[p, ti] = p_y[p, ti - 1] + icy 
            p_z[p, ti] = p_z[p, ti - 1] + icz 
    return p_x, p_y, p_z 

@jit(nopython = True, cache = True)
def square_boundaries(px , py, pz, incx, incy, incz, min_x, min_y, min_z, max_x, max_y, max_z): 
    """ 
    Defines a square simulation boundary. If the next particle position is out-of-bounds,
    we reverse the sign of the increment so the particle is inside the defined boundaries. 
    """
    num_p = px.shape[0] 
    ti_p = px.shape[1] 

    if px < min_x or px > max_x: 
        pcx = px - incx 

    if py < min_y or py > max_y:
        pcy = py - incy 

    if pz < min_z or pz > max_z:
        pcz = pz - incz 

    return pcx, pcy, pcz 

@jit(nopython = True , cache = True) 
def calculate_tamsd(pos_x, pos_y, pos_z):
    """ 
    Calculates TAMSD (time averaged mean squared displacement) of particle trajectories. 
    """
    particles = pos_x.shape[0]
    N = pos_x.shape[1] 
    tamsd = np.zeros(shape = (particles, N - 1)) 

    for p in np.arange(start = 0, stop = particles, step = 1): 
        for n in np.arange(start = 0, stop = N, step = 1): 
            sumdis = np.array([((pos_x[p, i + n] - pos_x[p, i]) ** 2 + (pos_y[p, i + n] - pos_y[p, i]) ** 2 + (pos_z[p, i + n] - pos_z[p, i]) ** 2) for i in np.arange(start = 1, stop = N - n, step = 1)]).sum()
            tamsd[p, n] = sumdis / (N - n) 
    return tamsd 

def plot_tamsd(dt, msd, label): 
    """
    Plots TAMSD  of particle trajectories. 
    """
    fig, ax = plt.subplots(1, 2, figsize = (8, 8))
    av_msd = np.mean(msd, axis = 0)

    for p in np.arange(0, msd.shape[0], step = 1):
        for t in np.arange(0, msd.shape[1], step = 1): 
            ax[0].plot(t, msd[p, t], 'bx')
            ax[1].plot(t, av_msd[t], 'ro')
    ax[0].set_xlabel('Time lag')
    ax[0].set_ylabel('TAMSD [pix^2]')
    ax[0].set_title('Individual TAMSDs: H = ' + str(label))
    ax[1].set_xlabel('Time lag')
    ax[1].set_ylabel('TAMSD [pix^2]')
    ax[1].set_title('Averaged TAMSDs: H = ' + str(label)) 
    ax[0].set_ylim([0, 30e3])

@jit(nopython = True, cache = True) 
def calculate_mss(pos_x, pos_y, pos_z):
    """ 
    Calculates MSS (moment scaling spectrum) of particle trajectories. 
    """
    particles = pos_x.shape[0]
    N = pos_x.shape[1] 
    mss = np.zeros(shape = (particles, N - 1)) 

    for p in np.arange(start = 0, stop = particles, step = 1): 
        for n in np.arange(start = 0, stop = N, step = 1): 
            sumdis = np.array([((pos_x[p, i + n] - pos_x[p, i]) ** 2 + (pos_y[p, i + n] - pos_y[p, i]) ** 2 + (pos_z[p, i + n] - pos_z[p, i]) ** 2) for i in np.arange(start = 1, stop = N - n, step = 1)]).sum()
            mss[p, n] = sumdis / (N - n) 
    return mss

def plot_mss(dt, mss): 
    """
    Plots MSS  of particle trajectories. 
    """
    print("MSS")

def plot_results_3d(p_x, p_y, p_z): 
    fig = plt.figure(figsize = (10, 10))
    ax3d = plt.axes(projection = '3d') 
    
    for p in np.arange(0, p_x.shape[0], step = 1): 
        for t in np.arange(0, p_x.shape[1], step = 1): 
            ax3d.plot3D(p_x[p, t], p_y[p, t], p_z[p, t], 'o') 

    ax3d.set_xlabel('X') 
    ax3d.set_ylabel('Y') 
    ax3d.set_zlabel('Z') 
    ax3d.set_title('3D particle trajectories')

def plot_results_2d(p_1, p_2, d_1 = 'X', d_2 = 'Y'): 
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes() 
    
    for p in np.arange(0, p_1.shape[0], step = 1): 
        for t in np.arange(0, p_1.shape[1], step = 1): 
            plt.plot(p_1[p, t], p_2[p, t], 'rx')
    ax.grid(b = 'True', which = 'major')
    ax.set_xlabel(d_1)
    ax.set_ylabel(d_2)
    ax.set_title('2D particle trajectories')

def plot_results_traj_3d(p_1, p_2, p_3, xmin, xmax, ymin, ymax, zmin, zmax): 
    fig, ax = plt.subplots(2 , 2, figsize = (10, 10))
    
    for p in np.arange(0, p_1.shape[0], step = 1): 
        for t in np.arange(0, p_1.shape[1], step = 1): 
            ax[0,0].plot(t, p_1[p, t], 'rx')    
            ax[0,1].plot(t, p_2[p, t], 'gx') 
            ax[1,0].plot(t, p_3[p, t], 'bx') 
            ax[1,1].plot(t, p_1[p, t], 'rx') 
            ax[1,1].plot(t, p_2[p, t], 'gx') 
            ax[1,1].plot(t, p_3[p, t], 'bx') 
    for a in ax.flat: 
        a.set(xlabel = 'Time steps', ylabel = 'Position')
    ax[0,0].set_title('X') 
    ax[0,0].set_ylim([xmin, xmax]) 
    ax[0,1].set_title('Y') 
    ax[0,1].set_ylim([ymin, ymax]) 
    ax[1,0].set_title('Z') 
    ax[1,0].set_ylim([zmin, zmax])
    ax[1,1].set_title('Positions combined') 
    ax[1,1].set_ylim([zmin, zmax])

def save_trajectories():
    """

    """

    return None
##################### Functions END ##################### 

# Parameters of the simulation 
num_part = args.num_part
x_dim = args.dim_x 
y_dim = args.dim_y
z_dim = args.dim_z 

t0 = args.t_start
t1 = args.t_end
n_steps = args.num_steps

H = args.hurst_exp 
n = args.n_time

M = args.num_M
gamma_H = gamma(H + 0.5)

# Define origin of simulation 
origin = x_dim // 2 
x0 = origin 
y0 = origin 
z0 = origin 

# If we choose n = 1 we do normal Brownian diffusion, else do fBM 
if n == 1:
    D = 1
    t = np.arange(start = 0, stop = n_steps, step = 1) # simulation time 
    t_phys = np.linspace(start = t0, stop = t1, num = n_steps) # physical time 
    dt = (t1 - t0) / n_steps 
    sigma = np.sqrt(2 * dt * D) 
    
    p_x, p_y, p_z = simulate_brownian(num_part, dt, t.shape[0], x0, y0, z0, sigma, drift = False) 
    msd = calculate_tamsd(p_x, p_y, p_z) 

    #Plotting results
    plot_tamsd(dt, msd) 
    #plot_results_2d(p_x, p_z, d_1 = 'X', d_2 = 'Z') 
    plot_results_3d(p_x, p_y, p_z) 
    plot_results_traj_3d(p_x, p_y, p_z, np.min(p_x), np.max(p_x), np.min(p_y), np.max(p_y), np.min(p_z), np.max(p_z)) 
else: 
    frac_steps = n * (n_steps + M) 
    t = np.arange(start = 0, stop = frac_steps, step = 1) # simulation time 
    t_phys = np.linspace(start = t0, stop = t1, num = frac_steps) # physical time 
    dt_frac = (t1 - t0) / frac_steps 

    p_x_frac, p_y_frac, p_z_frac = simulate_fractionalbrownian(num_part, H, M, n, t, x0, y0, z0, gamma_H) 
    p_x_h, p_y_h, p_z_h = simulate_fractionalbrownian(num_part, H, M, n, t, x0, y0, z0, gamma(0.9+0.5)) 
    p_x, p_y, p_z = simulate_fractionalbrownian(num_part, H, M, n, t, x0, y0, z0, gamma(0.1+0.5)) 
    msd_frac = calculate_tamsd(p_x_frac, p_y_frac, p_z_frac) 
    msd_frac_low= calculate_tamsd(p_x, p_y, p_z) 
    msd_frac_high= calculate_tamsd(p_x_h, p_y_h, p_z_h) 

    # Plotting results 
    plot_tamsd(dt_frac, msd_frac_low, label = 0.1) 
    plot_tamsd(dt_frac, msd_frac, label = 0.5) 
    plot_tamsd(dt_frac, msd_frac_high, label = 0.9) 
    #plot_results_2d(p_x_frac, p_z_frac, d_1 = 'X', d_2 = 'Z') 
    #plot_results_3d(p_x_frac, p_y_frac, p_z_frac) 
    #plot_results_traj_3d(p_x_frac, p_y_frac, p_z_frac, np.min(p_x_frac), np.max(p_x_frac), np.min(p_y_frac), np.max(p_y_frac), np.min(p_z_frac), np.max(p_z_frac)) 

plt.show() 