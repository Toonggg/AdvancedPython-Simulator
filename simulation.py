# Importing modules 
#import matplotlib
#matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numba import jit 

num_part = 50
x_dim = 1024
y_dim = x_dim
z_dim = x_dim
t_0 = 0
t_end = 0.1
n_step = 500
dt = (t_end - t_0) / n_step 

D = 1
sigma = np.sqrt(2 * dt * D)

# With Numba off, it takes almost 1 minute to simulate everything 
# Reflective boundary conditions 
@jit(nopython = True, cache = True)
def square_bounds(px , py, pz):
    num_p = px.shape[0]
    size_p = px.shape[1]
    min_x = 0 
    min_y = 0 
    min_z = 0 
    
    max_x = x_dim
    max_y = y_dim
    max_z = z_dim

    for p in np.arange(0, num_p):
        for i in np.arange(0, size_p):
            if px[p, i] < min_x or px[p, i] > max_x:
                px[p, i] = px[p, i - 1]
            if py[p, i] < min_y or py[p, i] > max_y:
                py[p, i] = py[p, i - 1]
            if pz[p, i] < min_z or pz[p, i] > max_z:
                pz[p, i] = pz[p, i - 1] 

    return None 

def spherical_bounds():

    return None 

plotting = 0

origin = x_dim / 2
x_0 = origin
y_0 = origin
z_0 = origin
p_0 = np.array([x_0, y_0, z_0])

# Generate Brownian increments
increment_x = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, n_step))
increment_y = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, n_step))
increment_z = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, n_step))

p_x = np.zeros(shape = (num_part, n_step)) 
p_y = np.zeros(shape = (num_part, n_step)) 
p_z = np.zeros(shape = (num_part, n_step)) 

# Initial position of particle 
p_x[:, 0] = x_0 + 50 * np.random.random(size = (1, num_part))
p_y[:, 0] = y_0 + 50 * np.random.random(size = (1, num_part))
p_z[:, 0] = z_0 + 50 * np.random.random(size = (1, num_part))

t_vec = np.arange(start = t_0 + 1, stop = n_step, step = 1)

# Simulate 
@jit(nopython = True, cache = True)
def simulate(num_part, p_x, p_y, p_z, increment_x, increment_y, increment_z):
    for p in np.arange(0, num_part, step = 1):
        for t in t_vec:
            p_x[p, t] = p_x[p, t - 1] + 1 * increment_x[p, t] 
            p_y[p, t] = p_y[p, t - 1] + 1 * increment_y[p, t] 
            p_z[p, t] = p_z[p, t - 1] + 1 * increment_z[p, t] 
            square_bounds(p_x, p_y, p_z)
    return p_x, p_y, p_z

p_x, p_y, p_z = simulate(num_part, p_x, p_y, p_z, increment_x, increment_y, increment_z)

def plot_results(plotting, num_part, t_vec, p_x, p_y, p_z): 
    fig = plt.figure(1)
    ax3d = plt.axes(projection = '3d') 
    
    for p in np.arange(0, num_part, step = 1): 
        for t in t_vec:
            ax3d.plot3D(p_x[p, t], p_y[p, t], p_z[p, t], 'x') 
            #plt.plot(t, p_x[p, t], 'x')
    
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    plt.show()

if plotting == 1:
    plot_results(plotting, num_part, t_vec, p_x, p_y, p_z)