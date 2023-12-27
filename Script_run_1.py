import os
from main_run import main
from proj_gokul import print_file

EDP=[]

os.system("cd SIAM && make")
os.system("cd ..")
#os.system("rm ./layer_record/inpu")
EDP=main(EDP)

print_file(EDP)