######################################################
# Author Greeshma Sasikumar                          #
# Script for calc                                    #
# Calculate energy, area and latency                 #
# Input: energy,area and latency csv                 #
# Output: Total values                               #
######################################################
import numpy as np                                     # needed for arrays
import pandas as pd                                    # data frame

def Calc_values():
    total_area=0
    total_readenergy=0
    total_leakage = 0
    total_readlatency=0
    total_subarray_area = 0
    total_routing_area = 0
    total_ADC_area = 0
    total_Accum_area = 0
    total_other_area = 0
    total_Buffer_area = 0
    total_noc_area = 0
    total_nop_area = 0
    total_nop_driver_area = 0

    total_routing_energy = 0
    total_ADC_energy = 0
    total_Accum_energy = 0
    total_Array_energy = 0
    total_other_energy = 0
    total_Buffer_energy = 0
    total_noc_power = 0
    total_nop_power = 0
    total_nop_driver_energy = 0

    total_routing_latency = 0
    total_ADC_latency = 0
    total_Accum_latency = 0
    total_Array_latency = 0
    total_other_latency = 0
    total_Buffer_latency = 0
    total_noc_latency = 0
    total_nop_latency = 0
    total_nop_driver_latency=0

    dataset = 'Final_Results'
    chiplet_breakup_file_name ='/home2/pnalla2/FFT_v2/FFT_SIAM/to_interconnect/chiplet_breakup.csv'
    data = pd.read_csv(chiplet_breakup_file_name, header=None)
    data = data.to_numpy()
    chiplet_num=len(data)

    my_area = pd.read_csv('/home2/pnalla2/FFT_v2/FFT_SIAM/' + dataset + '/area_chiplet.csv',header=None)
    my_energy = pd.read_csv('/home2/pnalla2/FFT_v2/FFT_SIAM/' + dataset + '/Energy_chiplet.csv',header=None)
    my_latency = pd.read_csv('/home2/pnalla2/FFT_v2/FFT_SIAM/' + dataset + '/Latency_chiplet.csv',header=None)
    my_string = my_area.iloc[0:,1].values                    # features are in columns 0:59
    my_chiparea = my_area.iloc[0:,2].values
    #print('chip area',my_string)
    for x,y in zip(my_string,my_chiparea):
        if (x=='Chip Area'):
            total_area=total_area+ float(y)
        elif (x == 'Subarray Area'):
            total_subarray_area = total_subarray_area + float(y)
        elif (x == 'Total Within Tile Routing Area'):
            total_routing_area = total_routing_area + float(y)
        elif (x == 'Total ADC (or S/As and precharger for SRAM) Area'):
            total_ADC_area = total_ADC_area + float(y) 
        elif (x == 'Total Accumulation Area'):
            total_Accum_area = total_Accum_area + float(y) 
        elif (x == 'Total Other Peripheries Area'):
            total_other_area = total_other_area + float(y)
        elif (x == 'Total Buffer Area'):
            total_Buffer_area = total_Buffer_area + float(y)
        elif (x == 'Total NoC area is'):
            total_noc_area = total_noc_area + float(y)
        elif (x == 'Total NoP area is'):
            total_nop_area = total_nop_area + float(y)
        elif (x == 'Total NoP Driver area is'):
            total_nop_driver_area = total_nop_driver_area + float(y)
        else:
            pass
    print('*****************************************************')
    print('Total chip area: ',total_area,' um^2')
    print ('Total Subarray Area: ', total_subarray_area, 'um2')
    print ('Total Routing Area: ', total_routing_area, 'um2')
    print ('Total ADC Area: ', total_ADC_area, 'um2')
    print ('Total Accum Area: ', total_Accum_area, 'um2')
    print ('Total Other Area: ', total_other_area, 'um2')
    print ('Total Buffer Area: ', total_Buffer_area, 'um2')
    #print ('Total NoC Area: ', total_noc_area*1e+6, 'um2')
    #print ('Total NoP Area: ', total_nop_area, 'mm2')
    #print ('Total NoP Driver Area: ', total_nop_driver_area, 'um2')
    print ('Total Area: ', total_area + total_noc_area*1e+6 + total_nop_area*1e+6 + total_nop_driver_area, 'um2')
    print('***************************************************** \n')


    my_string2 = my_latency.iloc[0:,1].values                    # features are in columns 0:59
    my_readlatency = my_latency.iloc[0:,2].values
    #print('chip area',my_string)
    for x,y in zip(my_string2,my_readlatency):
        if(x=='Total readLatency'):
            total_readlatency=total_readlatency+float(y)
        elif(x=='Total Array Latency'):
            total_Array_latency = total_Array_latency + float(y)
        elif (x == 'Total Routing Latency'):
            total_routing_latency = total_routing_latency + float(y)
        elif (x == 'Total ADC Latency'):
            total_ADC_latency = total_ADC_latency + float(y)
        elif (x == 'Total Accumulation Latency'):
            total_Accum_latency = total_Accum_latency + float(y)
        elif (x == 'Total Other Peripheries Latency'):
            total_other_latency = total_other_latency + float(y) 
        elif (x == 'Total Buffer Latency'):
            total_Buffer_latency = total_Buffer_latency + float(y)
        elif (x == 'Total NoC latency is'):
            total_noc_latency = total_noc_latency + float(y) 
        elif (x == 'Total NoP latency is'):
            total_nop_latency = total_nop_latency + float(y)
        elif (x == 'Total NoP Driver Latency is'):
            total_nop_driver_latency = total_nop_driver_latency + float(y)
        else:
            pass
    
    print('*****************************************************')
    print('Total read latency:',total_readlatency,'ns')
    print ('Total Array latency: ', total_Array_latency, 'ns')
    print ('Total Routing latency: ', total_routing_latency, 'ns')
    print ('Total ADC latency: ', total_ADC_latency, 'ns')
    print ('Total Accum latency: ', total_Accum_latency, 'ns')
    print ('Total Other latency: ', total_other_latency, 'ns')
    print ('Total Buffer latency: ', total_Buffer_latency, 'ns')
    #print ('Total NoC latency: ', total_noc_latency*1e9, 'ns')
    #print ('Total NoP latency: ', total_nop_latency*4*1e9, 'ns')
    #print ('Total NoP driver latency: ', total_nop_driver_latency, 'ns')
    print ('Total Chip latency: ', total_noc_latency*1e9+total_readlatency, 'ns')
    print('***************************************************** \n')

    my_string1 = my_energy.iloc[0:,1].values                    # features are in columns 0:59
    my_readenergy = my_energy.iloc[0:,2].values
    #print('chip area',my_string)
    for x,y in zip(my_string1,my_readenergy):
        if(x=='Total readEnergy'):
            total_readenergy=total_readenergy+ float(y)
        elif(x=='Total Array Energy'):
            total_Array_energy = total_Array_energy + float(y)
        elif (x == 'Total Routing Energy'):
            total_routing_energy = total_routing_energy + float(y)
        elif (x == 'Total ADC Energy'):
            total_ADC_energy = total_ADC_energy + float(y)
        elif (x == 'Total Accumulation Energy'):
            total_Accum_energy = total_Accum_energy + float(y)
        elif (x == 'Total Other Peripheries Energy'):
            total_other_energy = total_other_energy + float(y)
        elif (x == 'Total Buffer Energy'):
            total_Buffer_energy = total_Buffer_energy + float(y)
        elif (x == 'Total NoC power is'):
            total_noc_power = total_noc_power + float(y)
        elif (x == 'Total NoP power is'):
            total_nop_power = total_nop_power + float(y)
        elif (x == 'Total NoP Driver Energy is'):
            total_nop_driver_energy = total_nop_driver_energy + float(0)
        else:
            pass
    for x,y in zip(my_string1,my_readenergy):
        if(x=='Total leakage Energy'):
            total_leakage=total_leakage+ float(y)
    print('Total Leakage energy:',total_leakage,'pJ')

    print('*****************************************************')
    print('Total read energy:',total_readenergy,'pJ')
    print ('Total Array energy: ', total_Array_energy, 'pJ')
    print ('Total Routing energy: ', total_routing_energy, 'pJ')
    print ('Total ADC energy: ', total_ADC_energy, 'pJ')
    print ('Total Accum energy: ', total_Accum_energy, 'pJ')
    print ('Total Other energy: ', total_other_energy, 'pJ')
    print ('Total Buffer energy: ', total_Buffer_energy, 'pJ')
    #print ('Total NoC energy: ', total_noc_power/chiplet_num*total_noc_latency*1e9, 'pJ')
    #print ('Total NoP Power: ', total_nop_power, 'mW')
    #print ('Total NoP driver energy: ', total_nop_driver_energy, 'pJ')
    print ('Total energy: ', total_readenergy+total_noc_power/chiplet_num*total_noc_latency*1e9+total_leakage, 'pJ')
    print('***************************************************** \n')

if __name__ == "__main__":
    Calc_values()
