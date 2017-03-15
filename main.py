# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:28:03 2017

@author: jtc9242
"""
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from jinja2 import Template
import collections


# SETUP
MAX_PLYRS = 128

# GPU
# Create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

# Open kernel template code
tpl = Template(open("kernel.cu", "r").read())

# Render kernel from template and maximum number of players
rendered_tpl = tpl.render(
    type_name="float", MAX_PLYRS=MAX_PLYRS)

mod = SourceModule(rendered_tpl, no_extern_c=True)
gpujob = mod.get_function("gpuMain")


# Main function, using SourceModule
def run_function(n_players, probability, blocks=1024, block_size=1024):
    global gpujob
    start.record()  # Start timing

    # GPU settings
    nbr_values = blocks*block_size
    print "Using nbr_values =", nbr_values
    
    # Destination array that will receive the result
    wnrs = np.ones(nbr_values).astype(np.int32)

    gpujob(drv.Out(wnrs), np.int32(n_players), np.float32(probability), grid=(blocks, 1), block=(block_size, 1, 1))
    
    end.record()  # End timing
    # Calculate the run length
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print "SourceModule time: " + str(secs)
    return wnrs


# ANALYSIS
step_size = 0.025
probs = np.arange(step_size, 1.0+step_size, step_size)

for n_plyrs in range(2, MAX_PLYRS+1):
    data = []
    
    for prob in probs:
        n_blocks = 512
        blocksize = 1024
        n_games = n_blocks*blocksize
        
        print "Calculating for success probability "+str(prob)
        
        winners = list(run_function(n_plyrs, prob, blocks=n_blocks, block_size=blocksize))
        histogram = collections.Counter(winners)
        
        # Full weights
        weights = [0 for n in range(n_plyrs)]
        for i in histogram:
            weights[i] = float(histogram[i])/n_games
    
        data.append(weights)
    
    # Save data
    filename = 'data/'+str(n_plyrs)+'_plyrs'
    np.save(filename+'.npy', data)
