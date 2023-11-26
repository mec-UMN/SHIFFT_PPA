# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:22:09 2021

@author: skmandal
"""


import os, re, glob, sys, math
import timeit

from Interconnect.generate_traces_nop import generate_traces_nop
from Interconnect.run_booksim_mesh_chiplet_nop import run_booksim_mesh_chiplet_nop

start = timeit.default_timer()

# chiplet_size = 25
# num_chiplet = 144
# scale = 1
# bus_width = 4
# netname = 'VGG19_homogeneous_NoP_traces'

def nop_interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale):
    
    generate_traces_nop(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)

    print('Trace generation for NoP is finished')
    print('Starting to simulate the NoP trace')
    
    trace_directory_name = str(type) + '_' + str(num_chiplets) + '_cnt_size_' + str(chiplet_size) + '_scale_' + str(scale) + '_bus_width_' + str(bus_width)
    trace_directory_full_path = '/home2/pnalla2/FFT_v2/FFT_SIAM/Interconnect/' + netname + '_NoP_traces' + '/' + trace_directory_name
    
    results_directory_name = 'results_' + trace_directory_name
    results_directory_full_path = '/home2/pnalla2/FFT_v2/FFT_SIAM/Final_Results/NoP_Results_' + 'results_' + netname + '/' + results_directory_name
    
    os.system('pwd')
    
    # os.system('python3 run_booksim_mesh_chiplet_nop.py ' + trace_directory_full_path + ' ' + str(bus_width))
    run_booksim_mesh_chiplet_nop(trace_directory_full_path, bus_width)
    
    if (not os.path.exists(results_directory_full_path)):
    	os.makedirs(results_directory_full_path)
    
    
    # os.system('mv /home/gkrish19/SIAM_Integration/Interconnect/logs_NoP/ ' + results_directory_full_path)
    		
    
                    
