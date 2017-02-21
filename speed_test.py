# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:28:03 2017

@author: jtc9242
"""
import r
import random

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

n_players = 69
probability = 0.98


blocks = 1024
block_size = 1024
nbr_values = blocks*block_size

print "Using nbr_values ==", nbr_values

###GPU###

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

######################
# SourceModele SECTION
# We write the C code and the indexing and we have lots of control

mod = SourceModule(open("kernel.c", "r").read(), no_extern_c=True)
#mod = SourceModule(open("kernal_test.c", "r").read(), no_extern_c=True)

gpujob = mod.get_function("gpuMain")

# create a destination array that will receive the result
winners = numpy.ones(nbr_values).astype(numpy.int32)

start.record() # start timing

gpujob(drv.Out(winners), numpy.int32(n_players), numpy.float32(probability), grid=(blocks,1), block=(block_size,1,1) )

end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print "SourceModule time and first three results:"
print "%fs, %s" % (secs, str(winners[:3]))



###CPU###
def josephus_cpu(n_ply, prob):

    plyrs = [0] * n_players
    #C: int myArray[n_players] = { 0 }; // all elements 0
    
    survivors = len(plyrs) #survivors = sizeof(plyrs)/sizeof(int)
    while survivors > 1:
        #C: for(i = 0; i < sizeof(plyrs)/sizeof(int); i++)
        for i in range(0,len(plyrs)): #For all positions around the circle
            if plyrs[i] != 1: #If player in position i is not dead
                has_killed = False
                next_space=1 #Start by trying the player immediately to the right
                while has_killed == False:
                    #Calculate position of next player (accounting for looping around)
                    if i+next_space >= len(plyrs): #C: if i+next_space+1 >= sizeof(plyrs)/sizeof(int)
                        next_player_position = i+next_space - len(plyrs) #Calculate looped position
                    else:
                        next_player_position = i+next_space
                    if plyrs[next_player_position] != 1: #if next player is alive
                        dice_roll=random.random()
                        if dice_roll <=prob:
                            plyrs[next_player_position] = 1 #kill player
                            survivors-=1 #take one survivor off of survivor count
                        has_killed = True #break loop
                    else: #if next player is dead already
                        next_space+=1 #Move onto next position around the circle


    return r.which(plyrs,0)[0]

winners_cpu=[]

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

start.record() # start timing
start.synchronize()

for i in range(nbr_values):
    winners_cpu.append(josephus_cpu(n_players, probability))
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print "CPU time and first three results:"
print "%fs, %s" % (secs, str(winners_cpu[:3]))
