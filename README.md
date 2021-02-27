# AdvancedPython-Simulator
A Python simulator that aims to realistically simulate diffusion of single-molecules (in cells). Development is being done on Mac OS X 10.15.7 using Python 3. 
To use the code, I advise using (Mini)conda and/or pip and installing any necessary dependencies this way. 

### Features
* 3D simulation of Brownian diffusion for multiple particles. 
* 3D simulation of fractional Brownian motion for multiple particles with Hurst exponent H. 
  * Subdiffusion: 0 < H < 0.5 
  * Normal diffusion: H = 0.5
  * Superdiffusion: 0.5 < H < 1.0 

### TO-DO
* Add calculations/plots showing MSD (Mean Squared Displacement).
* Add calculations/plots showing MSS (Moment Scaling Spectrum).
* Implement collision of particle(s). 

### References
* [mplot3d](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
* [numba](http://numba.pydata.org/numba-doc/latest/user/jit.html)
* Fractals, Jens Feder (1988). 
