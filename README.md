# CuShittyJosephus
PyCUDA optimized Monte Carlo solver for the stochastic Josephus problem ("shitty Josephus").

## Requirements
* CUDA
* PyCUDA

## Guide
* To generate data from scratch, run main.py. GPU 0 will be used by default.
* Data is stored into .npy Numpy arrays
* Run plot.py to import saved data files and generate plots.
* speed_test.py compares the speed of GPU vs CPU versions of the simulation. Not very useful.

## Existing data
* See 'data' folder for saved data and plots from the initial run.
