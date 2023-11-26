#NoC estimation of SIAM tool
import os, re, glob, sys, math, shutil
import numpy as np
import pandas as pd
from subprocess import call
from pathlib import Path
import math

from Interconnect.generate_traces_noc import generate_traces_noc
from Interconnect.run_booksim_noc import run_booksim_noc


#Take all below parameters as argument
# quantization_bit = 8
# bus_width = 32
# netname = 'VGG-19_45.3M'
# xbar_size = 256
# chiplet_size = 9
# num_chiplets = 144
# type = 'Homogeneous_Design'
# scale = 100



def interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale):

    generate_traces_noc(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)

    print('Trace generation for NoC is finished')
    print('Starting to simulate the NoC trace')


    trace_directory_name = type + str(num_chiplets) + '_cnt_size_' + str(chiplet_size) + '_scale_' + str(scale) + '/'
    trace_directory_full_path = '/home2/pnalla2/FFT_v2/FFT_SIAM/Interconnect/' + netname + '_NoC_traces' + '/' + trace_directory_name
    
    results_directory_name = trace_directory_name
    results_directory_full_path = '/home2/pnalla2/FFT_v2/FFT_SIAM/Final_Results/NoC_Results_' + netname + '/' + results_directory_name
                
    run_booksim_noc(trace_directory_full_path)
    if (not os.path.exists(results_directory_full_path)):
    	os.makedirs(results_directory_full_path)
    
    
    os.system('mv /home2/pnalla2/FFT_v2/FFT_SIAM/Interconnect/logs/ ' + results_directory_full_path)


# interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)
