import os

os.system("cd SIAM && make")
os.system("cd ..")
#os.system("rm ./layer_record/inpu")
os.system("python main_run.py")