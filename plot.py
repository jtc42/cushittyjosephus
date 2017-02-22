# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 16:39:08 2017

@author: jtc9242
"""
import numpy as np
import matplotlib.pyplot as plt

max_plyrs=128

n_plyr_set = range(2,max_plyrs+1)
files=['data/'+str(n)+"_plyrs.npy" for n in n_plyr_set]

step_size = 0.025

probs = np.arange(0,1.0+step_size,step_size) #One more than is used. Colormap fix.
       
for f in files:
    data = np.load(f)
    plyrs = range(0,len(data[0])+1) #List of players in the game. One more than is used. Colormap fix.
    
    #Plot data
    plt.figure()
    plt.pcolormesh(plyrs, probs, data, cmap='gnuplot')
    plt.title(f)
    
    plt.axis([0, len(plyrs)-1, step_size, probs[-1]])
    
    cb = plt.colorbar()
    cb.set_label('Probability of winning')
    plt.xlabel('Position around circle')
    plt.ylabel('Probability of kill success')
    
    plt.savefig(f[:-4]+'.png', dpi=300)
    
    plt.close()
    