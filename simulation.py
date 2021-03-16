# Importing modules 
import argparse
import numpy as np
import scipy
import h5py 
import os, sys
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
from scipy.special import gamma
from numba import jit
import tkinter

# (1) https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib

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

parser.add_argument('-plot', '--plot-fig', action = 'store_true', help = 'Display plots of simulations.') 
args = parser.parse_args()

##################### Functions BEGINNING #####################
@jit(nopython = True, cache = True) 
def simulate_brownian(num_part, dt, time_steps, x0, y0, z0, sigma, drift = False): 
    """ 
    Simulates 3D Brownian motion (BM). The length of the simulations depends on the number of time_steps. Linear drift dependent
    on the product of dt and a randomnly sampled number can be added to X, Y, and Z positions. 

    Parameters
    ----------
    num_part : int
        Number of particles to simulate. 
    time_steps : int 
        Number of steps to simulate. 
    dt : float
        The time of each simulate step. 
    x0 : int 
        Starting location for the particle(s) on the X-axis. 
    y0 : int 
        Starting location for the particle(s) on the Y-axis. 
    z0 : int 
        Starting location for the particle(s) on the Z-axis. 
    sigma : float
        Dimensionless scaling factor for the increments in X, Y, and Z. 
    drift : bool
        Boolean used to add a linear drift term to the simulated particle position(s). 

    Returns
    -------
    p_x : ndarray
        Array containing the simulated BM positions of the particle(s) in X. Dimensions: (num_part, time_steps). 
    p_y : ndarray
        Array containing the simulated BM positions of the particle(s) in Y. Dimensions: (num_part, time_steps). 
    p_z : ndarray
        Array containing the simulated BM positions of the particle(s) in Z. Dimensions: (num_part, time_steps). 
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
    increment_x = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, time_steps - 1)) 
    increment_y = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, time_steps - 1)) 
    increment_z = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, time_steps - 1)) 

    # Pre-allocation of memory for particle positions 
    p_x = np.zeros(shape = (num_part, time_steps - 1)) 
    p_y = np.zeros(shape = (num_part, time_steps - 1))
    p_z = np.zeros(shape = (num_part, time_steps - 1))

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
    Simulates 3D fractional Brownian motion (fBM). Compared to the simulation of Brownian motion with drift,
    there is no explicit way to control the diffusion coefficient in fBM. 

    Parameters
    ----------
    num_part : int
        Number of particles to simulate. 
    H : float
        Hurst exponent of the particle motion. 
    M : int
        Range covered in time. 
    n : int
        Division of integer time. 
    t : ndarray
        Array containing time steps. 
    x0 : int
        Starting location for the particle(s) on the X-axis. 
    y0 : int
        Starting location for the particle(s) on the Y-axis. 
    z0 : int
        Starting location for the particle(s) on the Z-axis. 
    gamma_H : float
        Constant prefactor dependent on H in front of the X, Y, and Z increments. 

    Returns
    -------
    p_x : ndarray
        Array containing the simulated fBM positions of the particle(s) in X. Dimensions are the same as t. 
    p_y : ndarray
        Array containing the simulated fBM positions of the particle(s) in Y. Dimensions are the same as t. 
    p_z : ndarray
        Array containing the simulated fBM positions of the particle(s) in Z. Dimensions are the same as t. 
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
    p_x[:, 0] = x0 + 10 * np.random.random(size = (1, num_part)) 
    p_y[:, 0] = y0 + 10 * np.random.random(size = (1, num_part)) 
    p_z[:, 0] = z0 + 10 * np.random.random(size = (1, num_part)) 
    
    for p in np.arange(0, num_part, step = 1): 
        for ti in np.arange(start = 1, stop = t.shape[0], step = 1): 

            s1_x = np.array([ ((i ** (H - 0.5)) * incx[p, 1 + ti - i]) for i in range(1, n + 1)]).sum() 
            s2_x = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incx[p, 1 + ti - n - i]) for i in range(1, 1 + n * (M - 1))]).sum() 
            s1_y = np.array([ ((i ** (H - 0.5)) * incy[p, 1 + ti - i]) for i in range(1, n + 1)]).sum() 
            s2_y = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incy[p, 1 + ti - n - i]) for i in range(1, 1 + n * (M - 1))]).sum() 
            s1_z = np.array([ ((i ** (H - 0.5)) * incz[p, 1 + ti - i]) for i in range(1, n + 1)]).sum() 
            s2_z = np.array([ (((n + i) ** (H - 0.5) - i ** (H - 0.5)) * incz[p, 1 + ti - n - i]) for i in range(1, 1 + n * (M - 1))]).sum() 

            icx = gamma_H * (s1_x + s2_x) 
            icy = gamma_H * (s1_y + s2_y) 
            icz = gamma_H * (s1_z + s2_z) 

            p_x[p, ti] = p_x[p, ti - 1] + icx 
            p_y[p, ti] = p_y[p, ti - 1] + icy 
            p_z[p, ti] = p_z[p, ti - 1] + icz 
    return p_x, p_y, p_z 

@jit(nopython = True, cache = True)
def square_boundaries(px , py, pz, incx, incy, incz, min_x, min_y, min_z, max_x, max_y, max_z): 
    """ 
    Defines a square simulation boundary. If the next particle position(s) is/are out-of-bounds, 
    we reverse the sign of the increment(s) so the particle(s) is/are inside the defined boundaries. This check is performed each time step. 

    Parameters
    ----------
    px : float 
        Position of the particle(s) in X at a certain time step. 
    py : float 
        Position of the particle(s) in Y at a certain time step.
    pz : float 
        Position of the particle(s) in Z at a certain time step.
    incx : float
        Increment in X at a certain time step. 
    incy : float
        Increment in Y at a certain time step. 
    incz : float
        Increment in Z at a certain time step. 
    min_x : int
        Minimum value in X of the simulation box. 
    min_y : int
        Minimum value in Y of the simulation box. 
    min_z : int
        Minimum value in Z of the simulation box. 
    max_x : int
        Maximum value in X of the simulation box. 
    max_y : int
        Maximum value in Y of the simulation box. 
    max_z : int
        Maximum value in Z of the simulation box. 

    Returns
    -------
    pcx : float
        Corrected position(s) of the particle(s) in X at a certain time step. 
    pcy : float 
        Corrected position(s) of the particle(s) in Y at a certain time step. 
    pcz : float 
        Corrected position(s) of the particle(s) in Z at a certain time step. 
    """

    if px < min_x or px > max_x: 
        pcx = px - incx 

    if py < min_y or py > max_y:
        pcy = py - incy 

    if pz < min_z or pz > max_z:
        pcz = pz - incz 

    return pcx, pcy, pcz 

@jit(nopython = True , cache = True) 
def calc_msd(pos_x, pos_y, pos_z):
    """ 
    Calculates MSD (mean squared displacement) of particle trajectories with overlapping displacements. 

    Parameters
    ----------
    pos_x : ndarray 
        Particle positions in X. 
    pos_y : ndarray 
        Particle positions in Y. 
    pos_z : ndarray 
        Particle positions in Z.

    Returns
    -------
    msd : ndarray 
        MSD of the particle(s). 
    """
    particles = pos_x.shape[0]
    N = pos_x.shape[1] 
    tamsd = np.zeros(shape = (particles, N - 1)) 

    for p in np.arange(start = 0, stop = particles, step = 1): 
        for n in np.arange(start = 1, stop = N, step = 1): 
            sumdis = np.array([((pos_x[p, i + n] - pos_x[p, i]) ** 2 + (pos_y[p, i + n] - pos_y[p, i]) ** 2 + (pos_z[p, i + n] - pos_z[p, i]) ** 2) for i in np.arange(start = 1, stop = N - n, step = 1)]).sum()
            tamsd[p, n] = sumdis / (N - n) 
    return tamsd 

def plot_msd(msd, h_exp): 
    """
    Plots the MSD of particle(s). Both the individual MSDs and an average MSD are plotted. 
    A label specifying the Hurst exponent is also needed. 

    Parameters
    ----------
    msd : ndarray 
        First number to be subtracted. 
    h_exp : float
        Hurst exponent of the simulated particle(s). 
    """
    fig, ax = plt.subplots(1, 2, figsize = (10, 10))
    av_msd = np.mean(msd, axis = 0)

    for p in np.arange(0, msd.shape[0], step = 1):
        for t in np.arange(0, msd.shape[1], step = 1): 
            ax[0].plot(t, msd[p, t], 'bx')
            ax[1].plot(t, av_msd[t], 'ro')
    ax[0].set_xlabel('Time lag (number of steps)')
    ax[0].set_ylabel('MSD (pix^2)')
    ax[0].set_title('Individual TAMSDs: H = ' + str(h_exp))
    ax[1].set_xlabel('Time lag (number of steps)')
    ax[1].set_ylabel('MSD (pix^2)')
    ax[1].set_title('Averaged TAMSDs: H = ' + str(h_exp)) 
    ax[0].set_xlim([0, np.max(t)])
    ax[1].set_xlim([0, np.max(t)])
    ax[0].set_ylim([0, np.max(msd)]) 
    ax[1].set_ylim([0, np.max(av_msd)])

def plot_results_3d(p_x, p_y, p_z, h_exp = 0.5): 
    """
    Plots a 3D view of the positions of the particle(s) centered around the origin. 
    
    Parameters
    ----------
    p_x : ndarray
        Particle positions to plot in X. 
    p_y : ndarray
        Particle positions to plot in Y. 
    p_z : ndarray
        Particle positions to plot in Z. 
    h_exp : ndarray
        Hurst exponent of the simulated particle position(s).  
    """
    plt.figure(figsize = (10, 10))
    ax3d = plt.axes(projection = '3d') 

    color=iter(cm.rainbow(np.linspace(0,1,p_x.shape[0]))) # (1)
    labels = ['Particle ' + str(pl+1)  for pl in np.arange(0, p_x.shape[0], step = 1)]
    
    for p in np.arange(0, p_x.shape[0], step = 1): 
        c = next(color) # (1)
        for t in np.arange(0, p_x.shape[1], step = 1): 
            ax3d.plot3D(p_x[p, t], p_y[p, t], p_z[p, t], 'x', c = c, label = labels[p]) 
    legend_without_duplicate_labels(ax3d)
    ax3d.set_xlabel('X (pixels)') 
    ax3d.set_ylabel('Y (pixels') 
    ax3d.set_zlabel('Z (pixels)') 
    ax3d.set_xlim([origin-150,origin+150])
    ax3d.set_ylim([origin-150,origin+150])
    ax3d.set_zlim([origin-150,origin+150])
    ax3d.set_title('3D particle trajectories - H = ' + str(h_exp))

def legend_without_duplicate_labels(ax):
    """
    Drawing a plot legend without duplicate entries. Take from: 
    https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib 

    Parameters 
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object with duplicate legend entries. 
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique)) 

def plot_results_2d(p_1, p_2, d_1 = 'X', d_2 = 'Y'): 
    """
    Plots a 2D view of the particle(s). Two string labels d_1 and d_2 denoting the axis of the input arrays 
    are also provided. 

    Parameters  
    ----------
    p_1 : ndarray
        Array containing particle positions in X, Y, or Z. 
    p_2 : ndarray
        Array containing particle positions in X, Y, or Z. 
    d_1 : str
        String label for p_1. 
    d_2 : str 
        String label for p_2. 
    """
    plt.figure(figsize = (10, 10))
    ax = plt.axes() 

    color=iter(cm.rainbow(np.linspace(0,1,p_1.shape[0]))) # (1)
    labels = ['Particle ' + str(pl+1)  for pl in np.arange(0, p_1.shape[0], step = 1)]

    for p in np.arange(0, p_1.shape[0], step = 1): 
        c = next(color) # (c)
        for t in np.arange(0, p_1.shape[1], step = 1): 
            plt.plot(p_1[p, t], p_2[p, t], 'x', c = c, label = labels[p])
    legend_without_duplicate_labels(ax)
    ax.grid(b = 'True', which = 'major')
    ax.set_xlabel(d_1) 
    ax.set_ylabel(d_2)
    ax.set_title('2D particle trajectories')

def plot_results_traj_3d(p_x, p_y, p_z, xmin, xmax, ymin, ymax, zmin, zmax): 
    """
    Plots the X,Y, and Z positions of the particle(s) in 3 separate subplots and a combination subplot. 
    The axes limits can also be changed to plot a different range in both X and Y. 

    Parameters
    ----------
    p_x : ndarray 
        Array containing particle positions in X.
    p_y : ndarray
        Array containing particle positions in Y. 
    p_z : ndarray
        Array containing particle positions in Z. 
    xmin : int
        Smallest value to plot in X. 
    xmax : int
        Largest value to plot in X. 
    ymin : int
        Smallest value to plot in Y. 
    ymax : int
        Largest value to plot in Y. 
    zmin : int
        Smallest value to plot in Z. 
    zmax : int
        Largest value to plot in Z. 

    """
    fig, ax = plt.subplots(2 , 2, figsize = (10, 10))
    
    for p in np.arange(0, p_x.shape[0], step = 1): 
        for t in np.arange(0, p_x.shape[1], step = 1): 
            ax[0,0].plot(t, p_x[p, t], 'rx')    
            ax[0,1].plot(t, p_y[p, t], 'gx') 
            ax[1,0].plot(t, p_z[p, t], 'bx') 
            ax[1,1].plot(t, p_x[p, t], 'rx') 
            ax[1,1].plot(t, p_y[p, t], 'gx') 
            ax[1,1].plot(t, p_z[p, t], 'bx') 
    for a in ax.flat: 
        a.set(xlabel = 'Time steps', ylabel = 'Position')
    ax[0,0].set_title('X (pix)') 
    ax[0,0].set_ylim([xmin, xmax]) 
    ax[0,1].set_title('Y (pix)') 
    ax[0,1].set_ylim([ymin, ymax]) 
    ax[1,0].set_title('Z (pix)') 
    ax[1,0].set_ylim([zmin, zmax])
    ax[1,1].set_title('Positions combined') 
    ax[1,1].set_ylim([np.array([xmin, ymin, zmin]).min(), np.array([xmax, ymax, zmax]).max()])
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
    msd = calc_msd(p_x, p_y, p_z) 

    if args.plot_fig: 
        plot_msd(msd, h_exp = 0.5) 
        plot_results_2d(p_x, p_y, d_1 = 'X', d_2 = 'Y') 
        plot_results_3d(p_x, p_y, p_z, h_exp = 0.5) 
        plot_results_traj_3d(p_x, p_y, p_z, np.min(p_x), np.max(p_x), np.min(p_y), np.max(p_y), np.min(p_z), np.max(p_z)) 
        plt.show() 
else: 
    frac_steps = n * (n_steps + M) 
    t = np.arange(start = 0, stop = frac_steps, step = 1) # simulation time 
    t_phys = np.linspace(start = t0, stop = t1, num = frac_steps) # physical time 
    dt_frac = (t1 - t0) / frac_steps 

    gamma_H = (n ** -H) / (gamma(H + 0.5))
    
    p_x_frac, p_y_frac, p_z_frac = simulate_fractionalbrownian(num_part, H, M, n, t, x0, y0, z0, gamma_H) 
    msd_frac = calc_msd(p_x_frac, p_y_frac, p_z_frac) 

    if args.plot_fig:
        plot_msd(msd_frac, h_exp = H) 
        plot_results_2d(p_x_frac, p_y_frac, d_1 = 'X (pix)', d_2 = 'Y (pix)') 
        plot_results_3d(p_x_frac, p_y_frac, p_z_frac, h_exp = H) 
        plot_results_traj_3d(p_x_frac, p_y_frac, p_z_frac, np.min(p_x_frac), np.max(p_x_frac), np.min(p_y_frac), np.max(p_y_frac), np.min(p_z_frac), np.max(p_z_frac)) 
        plt.show() 