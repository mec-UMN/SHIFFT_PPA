# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 08:38:16 2021

@author: skmandal
"""

import pandas as pd
import numpy as np
import math
import os
import shutil

#Take all below parameters as argument
#quantization_bit = 8
#bus_width = 32
#netname = 'VGG-19_45.3M'
#xbar_size = 256
#chiplet_size = 9
#num_chiplets = 144
#type = 'Homogeneous Design'
#scale = 100

def generate_traces_noc(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale):

    # directory_name = netname + '/' + type + '/' + str(num_chiplets) + '_Chiplets_' + str(chiplet_size) + '_Tiles/to_interconnect'
    directory_name = '/home2/pnalla2/FFT_v2/FFT_SIAM/to_interconnect'
    tiles_csv_file_name = directory_name + '/num_tiles_per_layer_chiplet.csv'
    num_tiles_each_layer = pd.read_csv(tiles_csv_file_name, header=None)
    num_tiles_each_layer = num_tiles_each_layer.to_numpy()
    num_tiles_each_layer = num_tiles_each_layer[:, 2]
    
    activation_csv_file_name = directory_name + '/ip_activation.csv'
    num_activations_per_layer = pd.read_csv(activation_csv_file_name, header=None)
    num_activations_per_layer = num_activations_per_layer.to_numpy()
    
    chiplet_breakup_file_name = directory_name + '/chiplet_breakup.csv'
    data = pd.read_csv(chiplet_breakup_file_name, header=None)
    data = data.to_numpy()

    peripherial_file_name = directory_name + '/chip_peripherial.csv'
    data_1 = pd.read_csv(peripherial_file_name, header=None)
    data_1 = data_1.to_numpy()
    
    tile_begin_array = data[:, 0]
    tile_end_array = data[:, 1]
    adder_peripherial=data_1[:,1]
    buffer_peripherial=data_1[:,3]
    bool_peripherials=1
    dir_name = '/home2/pnalla2/FFT_v2/FFT_SIAM/Interconnect/' + netname + '_NoC_traces' + '/' + type + str(num_chiplets) + '_cnt_size_' + str(chiplet_size) + '_scale_' + str(scale)
            
    
    if (os.path.isdir(dir_name)):
        shutil.rmtree(dir_name)
    
    os.makedirs(dir_name)
    os.chdir(dir_name)
    
    num_chiplets_used = len(tile_begin_array)
    mesh_size = np.zeros([num_chiplets_used, 1])
    
    for chiplet_idx in range(0, num_chiplets_used):
        
        
        begin_layer = tile_begin_array[chiplet_idx]
        # print("Begin Layer: ", begin_layer)
        end_layer = tile_end_array[chiplet_idx]
        # print("End Layer: ", end_layer)
        
        num_tiles_this_chiplet = sum(num_tiles_each_layer[begin_layer:end_layer+1])
        mesh_size[chiplet_idx] = math.ceil(math.sqrt(num_tiles_this_chiplet+adder_peripherial+buffer_peripherial))
        
        if (begin_layer == 0):
            first_tile_number = 0
        else:
            first_tile_number = sum(num_tiles_each_layer[0:begin_layer])
        # print("Begin Layer is: ", begin_layer)
        # print("first_tile_number: ", first_tile_number)
        chiplet_dir_name = 'Chiplet_' + str(chiplet_idx)
        
        os.mkdir(chiplet_dir_name)
        
        for layer_idx in range (begin_layer, end_layer+bool_peripherials):
            
            trace = np.array([[0,0,0]])
            timestamp = 1

            ip_activation_this_layer = num_activations_per_layer[layer_idx, 1]
            num_packets_this_layer = math.ceil(ip_activation_this_layer*quantization_bit/(bus_width))
            num_packets_this_layer = math.ceil(num_packets_this_layer/scale)
            
            if (layer_idx == 0):
                src_tile_begin = 0
            else:
                # src_tile_begin = sum(num_tiles_each_layer[0:layer_idx-1])
                src_tile_begin = sum(num_tiles_each_layer[0:layer_idx])
            
            src_tile_end = src_tile_begin + num_tiles_each_layer[layer_idx] - 1
            
            dest_tile_begin = src_tile_end + 1
            #dest_tile_end = dest_tile_begin + num_tiles_each_layer[layer_idx+1] - 1
            dest_tile_end=dest_tile_begin+bool_peripherials
            # Normalize the number to first_tile_number
            # print("src_tile_begin before subtraction with first tile number: ", src_tile_begin)
            # print("first_tile_number: ", first_tile_number)
            src_tile_begin = src_tile_begin - first_tile_number
            # print("src_tile_begin: ", src_tile_begin)
            src_tile_end = src_tile_end - first_tile_number
            dest_tile_begin = dest_tile_begin - first_tile_number
            dest_tile_end = dest_tile_end - first_tile_number
            
            for packet_idx in range(0, num_packets_this_layer):
                for dest_tile_idx in range(dest_tile_begin, dest_tile_end):
                    for src_tile_idx in range(src_tile_begin, src_tile_end+1):
                        trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)
                        
                    if (dest_tile_idx != dest_tile_end-1):
                        timestamp = timestamp + 1
                timestamp = timestamp + 1
            
            if (buffer_peripherial==1) and (adder_peripherial==1):
                ip_activation_this_layer = num_activations_per_layer[layer_idx, 3]
                num_packets_accum_buffer = math.ceil(ip_activation_this_layer*quantization_bit/(bus_width))
                num_packets_accum_buffer = math.ceil(num_packets_accum_buffer/scale)
                
                src_tile_begin = dest_tile_end-1
                src_tile_end=src_tile_begin
                dest_tile_begin = src_tile_end+1
                dest_tile_end = dest_tile_begin
                for packet_idx in range(0, num_packets_accum_buffer):
                    for dest_tile_idx in range(dest_tile_begin, dest_tile_end+1):
                        for src_tile_idx in range(src_tile_begin, src_tile_end+1):
                            trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)
                        
                        if (dest_tile_idx != dest_tile_end):
                            timestamp = timestamp + 1
                    timestamp = timestamp + 1
            
            filename = 'trace_file_layer_' + str(layer_idx) + '.txt'

            trace = np.delete(trace, 0, 0)
            os.chdir(chiplet_dir_name)
            np.savetxt(filename, trace, fmt='%i')
            os.chdir("..")
    np.savetxt('mesh_size.csv', mesh_size, fmt='%i')
    os.chdir("..")
    os.chdir("..")
