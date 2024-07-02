import os
import numpy as np
import fileinput
import re
from main_run import main
from proj_gokul import print_file
import os
import shutil

import os
import shutil

# Source and destination directories
source_directory = "./Outputs"
destination_directory = "./Outputs/PreviousOutputs"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Get a list of all files in the source directory
files_to_move = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

# Move each file to the destination directory
for file_name in files_to_move:
    source_path = os.path.join(source_directory, file_name)
    destination_path = os.path.join(destination_directory, file_name)
    shutil.move(source_path, destination_path)

print("Files moved successfully.")

# Predefined array of new parameter values

adc_bits=2**np.array([7, 6, 5, 4])
adc_bits=adc_bits.tolist()
count = len(adc_bits)
mode = 0
# Parameter name to be replaced
#parameter_name = "levelOutput"
if mode == 1:
    xbar_size_1 = [256, 128, 64]
    xbar_size_2 = [256, 128, 64]
    size_chiplet_1 = [1, 2, 4]
    count =3
elif mode == 2:
    xbar_size_1 = [512, 256, 128]
    xbar_size_2 = [512, 256, 128]
    size_chiplet_1 = [1, 2, 4]
    count =3
elif mode == 3:
    xbar_size_1 = [1024, 512, 256, 128, 64]
    xbar_size_2 = [1024, 512, 256, 128, 64]
    size_chiplet_1 = [1, 2, 4, 16, 64]
    count =5
elif mode == 4:
    xbar_size_1 = [2048, 1024, 512, 256, 128]
    xbar_size_2 = [2048, 1024, 512, 256, 128]
    size_chiplet_1 = [1, 2, 4, 16, 64]
    count =5
elif mode == 5:
    xbar_size_1 = [4096, 2048, 1024, 512, 256, 128]
    xbar_size_2 = [4096, 2048, 1024, 512, 256, 128]
    size_chiplet_1 = [1, 2, 4, 16, 64, 256]
elif mode == 6:
    cellBit = [1, 2, 3, 4]


if mode == 0:
    # Parameter name to be replaced
    #parameter_name = "levelOutput"
    parameters_to_update = {
        "levelOutput": adc_bits
        #"size_chiplet_1":size_chiplet_1,
        #"numRowSubArray_type1":xbar_size_1,
        #"numColSubArray_type1":xbar_size_2,
    }
elif mode ==6:
    parameters_to_update = {
    #"levelOutput": adc_bits
    #"size_chiplet_1":size_chiplet_1,
    #"numRowSubArray_type1":xbar_size_1,
    #"numColSubArray_type1":xbar_size_2,
    "cellBit": cellBit,
    }
else:
    parameters_to_update = {
    #"levelOutput": adc_bits
    "size_chiplet_1":size_chiplet_1,
    "numRowSubArray_type1":xbar_size_1,
    "numColSubArray_type1":xbar_size_2,
    }
EDP=[]
values=[]
# Path to the C++ file
cpp_file_path = "./SIAM/Param.cpp"

for i in range (count):
    for parameter_name, param_newvalue in parameters_to_update.items():
        with fileinput.FileInput(cpp_file_path, inplace=True) as file:
            for line in file:
                # Check if the line contains the parameter assignment
                if parameter_name in line:
                    # Use regular expression to find the current parameter value
                    match = re.search(rf"{parameter_name}\s*=\s*([\d.]+)\s*;", line)
                    if match:
                        current_value = match.group(1)

                        # Get the next value from the array
                        new_value = str(param_newvalue.pop(0))

                        # Replace the current parameter value with the new value
                        line = line.replace(f"{parameter_name} = {current_value}", f"{parameter_name} = {new_value}")

                print(line, end='')

    print(f"Updated values in '{cpp_file_path}'.")
     # Read the C++ file and replace the parameter value
    with fileinput.FileInput(cpp_file_path, inplace=True) as file:
        for line in file:
            # Check if the line contains the parameter assignment
            if parameter_name in line:
                # Use regular expression to find the current parameter value
                match = re.search(rf"{parameter_name}\s*=\s*([\d.]+)\s*;", line)
                if match:
                    values.append(match.group(1))
             # Print the line to the file
            print(line, end='')

    os.system("cd SIAM && make")
    os.system("cd ..")
    #os.system("rm ./layer_record/inpu")
    EDP=main(EDP)

print_file(EDP)
print(values)