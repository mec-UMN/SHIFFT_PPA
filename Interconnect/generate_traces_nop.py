# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:57:35 2021

@author: skmandal
"""

import pandas as pd
import numpy as np
import math
import os
import shutil

#Take all below parameters as argument
# quantization_bit = 8
# bus_width = 32
# netname = 'ResNet110' #'VGG-19_45.3M'
# xbar_size = 256
# chiplet_size = 16
# num_chiplets = 25
# type = 'Homogeneous Design'
# scale = 1

def generate_traces_nop(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale):

    # directory_name = netname + '/' + type + '/' + str(num_chiplets) + '_Chiplets_' + str(chiplet_size) + '_Tiles/to_interconnect'
    directory_name = '/home/nalla052/SHIFFT_PPA/to_interconnect'
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
    
    tile_begin_array = data[:, 0]
    tile_end_array = data[:, 1]
    
    num_chiplets_this_layer = data[:, 2]
    
    dir_name = '/home/nalla052/SHIFFT_PPA/Interconnect/' +  netname + '_NoP_traces' + '/' + type + '_' + str(num_chiplets) + '_cnt_size_' + str(chiplet_size) + '_scale_' + str(scale) + '_bus_width_' + str(bus_width)
            
    
    if (os.path.isdir(dir_name)):
        shutil.rmtree(dir_name)
    
    os.makedirs(dir_name)
    # os.chdir(dir_name);
    
    num_chiplets_used = sum(num_chiplets_this_layer)
    nop_mesh_size = math.ceil(math.sqrt(num_chiplets_used))
    
    num_nop_transactions = len(num_chiplets_this_layer)
    
    chiplet_idx = 0
    trace_file_idx = 0
    num_bits = 0
    
    for nop_trans_idx in range(0, num_nop_transactions-1):
        
        trace = np.array([[0,0,0]])
                    
        begin_layer_this_chiplet = tile_begin_array[nop_trans_idx]
        end_layer_this_chiplet = tile_end_array[nop_trans_idx]
        
        begin_layer_next_chiplet = tile_begin_array[nop_trans_idx + 1]
        end_layer_next_chiplet = tile_end_array[nop_trans_idx + 1]
        
        
        if (begin_layer_this_chiplet != end_layer_this_chiplet):
            num_src_chiplet = 1
        else:
            num_src_chiplet = num_chiplets_this_layer[nop_trans_idx]
        
        src_chiplet_begin = chiplet_idx
        src_chiplet_end = chiplet_idx + num_src_chiplet - 1
        
        
        if (begin_layer_next_chiplet != end_layer_next_chiplet):
            num_dst_chiplet = 1
        else:
            num_dst_chiplet = num_chiplets_this_layer[nop_trans_idx + 1]
        
        dst_chiplet_begin = src_chiplet_end + 1
        dst_chiplet_end = dst_chiplet_begin + num_dst_chiplet - 1
        
        
        num_bits = num_bits + num_activations_per_layer[begin_layer_next_chiplet]*quantization_bit
        num_activations_per_chiplet = math.ceil(num_activations_per_layer[begin_layer_next_chiplet]/(num_src_chiplet*num_dst_chiplet*scale*bus_width));
        
        
        
        chiplet_idx = dst_chiplet_begin
        
        
        
        
        timestamp = 1
        
        
        for packet_idx in range(1, num_activations_per_chiplet):
            for dest_chiplet_idx in range(dst_chiplet_begin, dst_chiplet_end+1):
                for src_chiplet_idx in range(src_chiplet_begin, src_chiplet_end+1):
                    # trace = [trace; src_chiplet_idx-1 dest_chiplet_idx-1 timestamp]
                    trace = np.append(trace, [[src_chiplet_idx, dest_chiplet_idx, timestamp]], axis=0)
                    
                if (dest_chiplet_idx != dst_chiplet_end):
                    timestamp = timestamp + 1
                
                
            
            timestamp = timestamp + 1
            
            
        filename = 'trace_file_chiplet_' + str(nop_trans_idx) + '.txt'
        
        trace = np.delete(trace, 0, 0)
        os.chdir(dir_name)
        np.savetxt(filename, trace, fmt='%i')
        os.chdir("../..")
    
    os.chdir(dir_name)
    np.savetxt('nop_mesh_size.csv', [nop_mesh_size], fmt='%i')
    os.chdir("../..")