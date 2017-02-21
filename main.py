# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:28:03 2017

@author: jtc9242
"""
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule


###GPU###

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

# SourceModele SECTION
mod = SourceModule(open("kernel.c", "r").read(), no_extern_c=True)
gpujob = mod.get_function("gpuMain")

def run_function(n_players, probability, blocks = 1024, block_size = 1024):
    global gpujob
    #GPU settings
    nbr_values = blocks*block_size
    print "Using nbr_values =", nbr_values
    
    # Destination array that will receive the result
    winners = np.ones(nbr_values).astype(np.int32)
    
    start.record() # start timing
    
    gpujob(drv.Out(winners), np.int32(n_players), np.float32(probability), grid=(blocks,1), block=(block_size,1,1) )
    
    end.record() # end timing
    # calculate the run length
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print "SourceModule time: " + str(secs)
    return winners
    



###ANALYSIS
import collections


#n_plyrs = 69
step_size = 0.025
probs = np.arange(step_size,1.0+step_size,step_size)

for n_plyrs in range(2,129):
    data=[]
    
    for prob in probs:
        blocks=512
        blocksize=1024
        n_games = blocks*blocksize
        
        print "Calculating for success probability "+str(prob)
        
        winners=list(run_function(n_plyrs, prob, blocks=blocks, block_size=blocksize))
        
        histogram=collections.Counter(winners)
        
        #Full weights
        weights = [0 for n in range(n_plyrs)]
        for i in histogram:
            weights[i] = float(histogram[i])/n_games
    
        data.append(weights)
    
    #Save data
    filename=str(n_plyrs)+'_plyrs'
    np.save(filename+'.npy', data)    