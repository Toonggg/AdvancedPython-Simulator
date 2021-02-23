# Importing modules 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
#import argparse
#import os, sys
#import numba, pyopencl

num_part = 1
x_dim = 1024
y_dim = x_dim
z_dim = x_dim
T_0 = 0
T_end = 200
T_step = 1000
dT = (T_end - T_0) / T_step

D = 1
sigma = np.sqrt(2 * dT * D)

origin = x_dim / 2
x_0 = origin
y_0 = origin
z_0 = origin
p_0 = np.array([x_0, y_0, z_0])

# Generate Brownian increments
increment_x = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, T_step))
increment_y = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, T_step))
increment_z = np.random.normal(loc = 0.0, scale = sigma, size = (num_part, T_step))

p_x = np.zeros(shape = (num_part, T_step))
p_y = np.zeros(shape = (num_part, T_step))
p_z = np.zeros(shape = (num_part, T_step))

# Initial position of particle 
p_x[0, 0] = x_0
p_y[0, 0] = y_0
p_z[0, 0] = z_0

# Simulate
for t in np.arange(start = T_0 + 1, stop = T_end, step = 1):
    p_x[0, t] = p_x[0, t - 1] + 3 * increment_x[0, t] 
    p_y[0, t] = p_y[0, t - 1] + 3 * increment_y[0, t] 
    p_z[0, t] = p_z[0, t - 1] + 1 * increment_z[0, t] 

fig = plt.figure(1)
ax3d = plt.axes(projection = '3d')
for t in np.arange(start = T_0 , stop = T_end, step = 1):
    ax3d.plot3D(p_x[0, t], p_y[0, t], p_z[0, t], 'rx') 

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
plt.show()