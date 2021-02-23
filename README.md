# AdvancedPython-Simulator
A Python simulator that aims to realistically simulated molecular diffusion of single-molecules in cells. To get things working, old Matlab
code will be used as a basis and expanded upon. The initial goal will be to simulate one single-particle very well. This can be extended to multiple particles.
Normal Brownian motion will initially be implemented, however different diffusion models (anomolous diffusion - using the Hurst exponent) will also be added. 
Drift in all the dimensions will also be added. Switching of the diffusive motion of particles is also to be implemented. Finally, nice visualization of the trajectories is also to be done. The code is supposed to be as modular as possible. 

### Features
* 3D simulation of Brownian motion for multiple particles (drift to be added later). 

### TO-DO
* Implement anomalous diffusion using Hurst exponent. 
* Add nice visualization of simulation (plots of 3D or 2D trajectories). 

### References
[mplot3d](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
[numba](http://numba.pydata.org/numba-doc/latest/user/jit.html)
