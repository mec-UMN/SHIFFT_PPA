#Author: Gokul Krishnan
#SIAM Tool Run File

import os
import numpy as np
from subprocess import call
from pathlib import Path
import time
import shutil
import pandas as pd
from Interconnect.noc_estimation import interconnect_estimation
from Interconnect.nop_estimation import nop_interconnect_estimation
from Interconnect.nop_estimation_big_little import nop_interconnect_estimation_big_little
#from NoP_hardware import *
from proj_gokul import Calc_values
from proj_gokul import print_file
import math
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

# Globabal variables for the Simulation
num_classes = 1

# NoC Parameters
quantization_bit = 8
weight_length = quantization_bit
input_length = quantization_bit
bus_width = 32
netname = 'FFT'
xbar_size = 128 # AAG Not used anywhere
chiplet_size = 32 # Number of IMC tiles inside chiplet # AAG Not used anywhere
num_chiplets = 1 # Total number of Chiplets used # AAG Not used anywhere
type = 'Homogeneous_Design'
scale = 10

# NoP Parameters - Extracted from GRS Nvidia 
#ebit = 0.54 # pJ @250MHz
#area_per_lane = 5304.5 #um2
#clocking_area = 10609 #um2
n_lane = 40


# NoP bus width for different type of chiplet
n_lane_big = 40
n_lane_little = 40
n_lane_mid=40

n_lane_list=[19,25,32,38,38,50,76,76,102] # in network.csv file "1" first type of chiplet "2" second type of chiplet

#n_bits_per_chiplet = 1047616 #Automate this in next version. enter total activation count and adjust the percentage inside the NoP hardware python file R110
scale_nop = 100
big_little_boundary = [10,10,12,15,15,15] #decide the nop buswidth depends on the layer number
freq_big = 1    # Frequency of big chiplets
freq_little = 1  # Frequency of little chiplets
# ratio = 0.94 # Ratio of the packets in little to the packets in big DenseNet-40
# ratio = 0.63 # VGG-19
#ratio = 0.58 # R110

def model_test(dataset, runs, data_path):
	gcs_utils._is_gcs_disabled = True
	if dataset == "crema_d":
		ds = tfds.load('crema_d', split='test', shuffle_files=True, data_dir=data_path, try_gcs=False, as_supervised=True)
	elif dataset == "speech_commands":
		ds = tfds.load('speech_commands', split='test', shuffle_files=True, data_dir=data_path, try_gcs=False, as_supervised=True)
	elif dataset == "groove":
		ds = tfds.load('groove/full-16000hz', split='test', shuffle_files=True, data_dir=data_path, try_gcs=False, as_supervised=False)
	elif dataset == "nsynth":
		ds = tfds.load('nsynth', split='test', shuffle_files=True, data_dir=data_path, try_gcs=False, as_supervised=False)
	else:
		assert False, "Unknow dataset : {}"
	assert isinstance(ds, tf.data.Dataset)
	print(ds)
	ds = ds.take(runs)  # Only take a single example
	#import pdb;pdb.set_trace()
	return ds

def weight_generate(size_x, size_y, length):
    weight = np.zeros([size_x, size_y*length], dtype='int8')
    for i in range (0, size_x):
        for j in range(0,size_x):
            weight_ele = np.array([np.cos(2*math.pi*i*j/size_x)])
            #import pdb;pdb.set_trace()
            weight[i, j*length:(j+1)*length] = dec2bin(weight_ele, length)[0]
            weight_ele = np.array([np.sin(2*math.pi*i*j/size_x)])
            weight[i, (size_x+j)*length:(size_x+j+1)*length] = dec2bin(weight_ele, length)[0]
        print(i)
    #import pdb;pdb.set_trace()
    return weight

def input_generate(size, k, in_width):
    input = np.zeros([size, in_width], dtype='int8')
    if k is None:
       for i in range (0, size):
           input[i] = 1
           #input[i] = random.randrange(1, 2**in_width)
    else:
        #import pdb;pdb.set_trace()
        sample_rate = k.size
        n = math.floor(sample_rate/size)
        arr = n*np.arange(size)
        input_unquantized = k[arr]
        print(input_unquantized)
        for i in range(0,size):
            #import pdb;pdb.set_trace()
            input[i,:] = dec2bin(np.array([input_unquantized[i]/(1.4*np.mean(np.absolute(input_unquantized)))]), in_width)[0]
    return input

def write_matrix_weight(weight_matrix, filename):
    #cout = input_matrix.shape[-1]
    #weight_matrix = input_matrix.reshape(-1, cout)
    import pdb;pdb.set_trace()
    np.savetxt(filename, weight_matrix, delimiter=",", fmt='%10.5f')

def write_matrix_input(input_matrix, length, filename):
    input_shape = input_matrix.shape
    """
    my_file = Path("./to_interconnect/ip_activation.csv")
    if my_file.is_file():
        with open('./to_interconnect/ip_activation.csv', 'a') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]))
            f.write("\n")
            f.close()
    else:
        with open('./to_interconnect/ip_activation.csv', 'w') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]))
            f.write("\n")
            f.close()
    
    cout = input_matrix.shape[-1]
    input_matrix_re = input_matrix.reshape(-1,cout)
    input_matrix_up = input_matrix_re
    for i in range(0, length-1):
        input_matrix_up = np.append(input_matrix_up, input_matrix_re,1)
    input_matrix_up[:,0] = 0
    """
    np.savetxt(filename, input_matrix, delimiter=",", fmt='%10.5f')

def write_matrix_activation_conv(input_matrix, fill_dimension, length, filename):
    filled_matrix_b = np.ones([input_matrix.shape[2], input_matrix.shape[1] * length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i::length] = b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')

def write_matrix_activation_fc(input_matrix, input_length, fill_dimension, length, filename):
    filled_matrix_b = np.ones([input_matrix.shape[1], length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i] = b
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')

def stretch_input(input_matrix, input_length, window_size = 5):
    input_shape = input_matrix.shape
    my_file = Path("./to_interconnect/ip_activation.csv")
    if my_file.is_file():
        with open('./to_interconnect/ip_activation.csv', 'a') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]*input_length))
            f.write("\n")
            f.close()
    else:
        with open('./to_interconnect/ip_activation.csv', 'w') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]*input_length))
            f.write("\n")
            f.close()
    # print("input_shape", input_shape)
    item_num_1 = ((input_shape[2]) - window_size + 1) * ((input_shape[3])-window_size + 1)
    item_num = max(item_num_1, 1)
    if (item_num_1==0):
        output_matrix = np.ones((input_shape[0],item_num,input_shape[1]*(window_size-1)*(window_size-1)))
    else:
        output_matrix = np.ones((input_shape[0],item_num,input_shape[1]*(window_size)*window_size))
    iter = 0
    for i in range( max(input_shape[2]-window_size + 1, 1) ):
        for j in range( max(input_shape[3]-window_size + 1, 1) ):
            for b in range(input_shape[0]):
                if (item_num_1==0):
                    output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size-1,j: j+window_size-1].reshape(input_shape[1]*(window_size-1)*(window_size-1))
                else:
                    #breakpoint()
                    output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1
    return output_matrix


def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta
    #import pdb;pdb.set_trace()
    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y[0].copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y[0].copy())
        scale_list.append(base * delta)
    out_arr=np.array(out)
    #out_re=out_arr.reshape(1, out_arr.size)
    #import pdb;pdb.set_trace()
    return out_arr,scale_list


def main(EDP):
    
    IN = []
    W = []
    print("Starting the parsing of the network")
    # delete directories if these exist
    
    write_files =True
    if not os.path.exists('./layer_record'):
        os.makedirs('./layer_record')
    else:
        dir_name = "./layer_record"
        test = os.listdir(dir_name)
        for item in test:
            if item.endswith(".csv"):
                #os.remove(os.path.join(dir_name, item))
                write_files = False
    output_path = './layer_record/'

    if os.path.exists('./layer_record/trace_command.sh'):
        os.remove('./layer_record/trace_command.sh')

    if os.path.exists('./to_interconnect'):
        shutil.rmtree('./to_interconnect')
    os.makedirs('./to_interconnect')
    
    if os.path.exists('./Final_Results'):
        shutil.rmtree('./Final_Results')
    os.makedirs('./Final_Results')

    f = open('./layer_record/trace_command.sh', "w")
    f.write('./SIAM/main ./SIAM/NetWork.csv '+str(weight_length)+' '+str(input_length)+' ')
    # Read the NetWork.csv file
    #network_params = np.loadtxt('./Networks/ResNet/110/NetWork.csv', dtype=int, delimiter=',')
    network_params = np.loadtxt('./SIAM/NetWork.csv', ndmin=2, dtype=int, delimiter=',')
    num_layers = network_params.shape[0]
    
    # extract every layer's type of chiplet
    layer_type=network_params[:,8]
    
    # Create input matrix and weight matrix
    # Ideally if should be extracted from the network itself frm Pytroch or TensorFlow. Need to add this.
    # In interest of the different sturctures we can have higher flexibility
    
    if write_files:
    # Regular Code
        for layer_idx in range(0, num_layers):            
            params_row = network_params[layer_idx]
            ds=model_test("speech_commands",1,'./dataset/')
            for audio,label in tfds.as_numpy(ds):
                #import pdb;pdb.set_trace()
                temp_array_IN =input_generate(network_params[layer_idx][3],  np.float64(audio), input_length)
            IN.append(temp_array_IN)
            """
            temp_array_IN = np.ones(shape=(1, network_params[layer_idx][2], \
                                            network_params[layer_idx][1], \
                                                network_params[layer_idx][0]), dtype='int8')
            IN.append(temp_array_IN)

            
            if (layer_idx < (num_layers)):
                    temp_array_W = np.ones(shape=(network_params[layer_idx][4], \
                                                        network_params[layer_idx][3], network_params[layer_idx][2], \
                                                            network_params[layer_idx][5]), dtype='int8')
            else:
                temp_array_W = np.ones(shape=(network_params[layer_idx][4], \
                                                    network_params[layer_idx][3], network_params[layer_idx][2], \
                                                        network_params[layer_idx][5]), dtype='int8')
            W.append(temp_array_W)
            """
        #f.write('./SIAM/main ./Networks/ResNet/110/NetWork.csv '+str(weight_length)+' '+str(input_length)+' ')
        # Debug Line
        # f.write('gdb --args ./SIAM/main ./SIAM/NetWork.csv '+str(weight_length)+' '+str(input_length)+' ')
            temp_array_W =weight_generate(network_params[layer_idx][3],network_params[layer_idx][5], weight_length)
            W.append(temp_array_W)
            import pdb;pdb.set_trace()
        for i,(input,weight) in enumerate(zip(IN,W)):
            input_file_name = 'input_layer' + str(i) + '.csv'
            weight_file_name = 'weight_layer' + str(i) + '.csv'
            f.write(output_path + weight_file_name+' '+output_path + input_file_name+' ')
            write_matrix_weight(weight, output_path + weight_file_name)
            write_matrix_input(input, input_length, output_path + input_file_name)
            """
            if len(weight.shape) > 2:
                k = weight.shape[0]
                write_matrix_activation_conv(stretch_input(input, input_length, k), None, input_length, output_path + input_file_name)
            else:
                write_matrix_activation_fc(input, input_length, None, input_length, output_path + input_file_name)
            """
    else:
        for i in range(0, num_layers): 
            input_file_name = 'input_layer' + str(i) + '.csv'
            weight_file_name = 'weight_layer' + str(i) + '.csv'
            f.write(output_path + weight_file_name+' '+output_path + input_file_name+' ')
    f.close()


    # # Estimation of computation performance
    print("Starting the Estimation of the Performance")
    start = time.time()
    
    call(["/bin/bash", "./layer_record/trace_command.sh"])
    # # start = time.time()
    
    # perform cycle accurate noc simulation
    #interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)
    
    # NoP Estimation
    #nop_interconnect_estimation_big_little(quantization_bit, n_lane_little, n_lane_big,n_lane_mid,n_lane_list,freq_big, freq_little, netname, xbar_size, chiplet_size, num_chiplets, type, scale_nop, big_little_boundary)

    # Calculate and Dump NoP Hardware Cost
    #network_params = np.loadtxt('./Networks/test/test_different_size_chiplet.csv', dtype=int, delimiter=',')
    #num_layers = network_params.shape[0]

    # extract every layer's type of chiplet
    layer_type=network_params[:,8]


    chiplet_breakup_file_name ='/home2/pnalla2/FFT_v2/FFT_SIAM/to_interconnect/chiplet_breakup.csv'
    data = pd.read_csv(chiplet_breakup_file_name, header=None)
    data = data.to_numpy()
    num_used_chiplet=data
    
    activation_breakup_file_name ='/home2/pnalla2/FFT_v2/FFT_SIAM/to_interconnect/ip_activation.csv'
    data = pd.read_csv(activation_breakup_file_name, header=None)
    activation_per_chiplet = data.to_numpy()

    """
    NoP_driver_area, NoP_driver_energy,NoP_driver_latency =0,0,0 #NoP_AIB_estimation(num_used_chiplet,layer_type,activation_per_chiplet,n_lane_list)
    #                                                

                         
    area_file = open('/home2/pnalla2/FFT_v2/FFT_SIAM/Final_Results/area_chiplet.csv', 'a')
    area_file.write(''+ ',' +'Total NoP Driver area is' + ',' + str(NoP_driver_area) + ',' + 'um^2')
    area_file.close()

    latency_file = open('/home2/pnalla2/FFT_v2/FFT_SIAM/Final_Results/Latency_chiplet.csv', 'a')
    latency_file.write(''+ ',' +'Total NoP Driver Latency is' + ',' + str(NoP_driver_latency) + ',' + 'ns')
    latency_file.close()

    energy_file = open('/home2/pnalla2/FFT_v2/FFT_SIAM/Final_Results/Energy_chiplet.csv', 'a')
    energy_file.write(''+ ',' +'Total NoP Driver Energy is' + ',' + str(NoP_driver_energy) + ',' + 'pJ')
    energy_file.close()
    """

    end = time.time()
    #print("The SIAM sim time is:", (end - start))
    EDP= Calc_values(EDP)
    return EDP

    
if __name__ == "__main__":
    main()
