from MeanShift import *
import numpy as np
import time

start_time = time.time()

# def __init__ (self,bandwidth,dimension,count,window_size,max_iterations,threshold):
meanshift = MeanShift(2,2,10,2,100,0.001)
# meanshift.visualize2d()
# meanshift.wireframe3d()
# meanshift.visualize3d()
# meanshift.printAllValues()

print("--- %s seconds ---" % (time.time() - start_time))