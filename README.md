# AdvancedPython-Simulator
A Python simulator that aims to realistically simulate diffusion of single-molecules (in cells). Development is being done on Mac OS X 10.15.7 using Python 3. 
Additional modules that will be needed are mentioned below. The simulation of 3D fractional Brownian motion (fBm) can be done using several different methods. Some details can be found in the [fBm wiki](https://en.wikipedia.org/wiki/Fractional_Brownian_motion). The implemented method in the simulations is based on a method detailed in the 1988 book by Jens Feder. 

### Features
* 3D simulation of Brownian diffusion for multiple particles. 
* 3D simulation of fBm for multiple particles with Hurst exponent H. 
  * Subdiffusion: 0 < H < 0.5 
  * Normal diffusion: H = 0.5
  * Superdiffusion: 0.5 < H < 1.0 

### TO-DO
* Organize code into several files to aid readability. 
* Implement simulation of continuous time random walk (CTRW). 
* Add calculations/plots showing MSS (Moment Scaling Spectrum).
* Implement collision of particle(s). 

### Dependencies
Some additional modules are needed to run the code. This list will change of course in the future as features are added that need different modules. 
For now, the current modules are needed. 
```
NumPy
SciPy
Matplotlib
Numba 
h5py
```

### References
* [Matplotlib](https://matplotlib.org/stable/index.html)
* [mplot3d](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
* [Numba](http://numba.pydata.org/numba-doc/latest/user/jit.html)
* [NumPy](https://numpy.org/)
* [SciPy](https://www.scipy.org/)
* [h5py](http://www.h5py.org/)
* Fractals, Jens Feder (1988). 
