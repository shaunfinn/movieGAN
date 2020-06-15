import os
import shutil
#current drectory should be script directory 
cache_dir = str(os.getcwd()) +"/__pycache__" 

#removes pycache if exists
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("cache removed")
else:
    print("no cache to remove")
