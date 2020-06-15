# copies current checkpoint.tar and config.py files to  model_dir

from config import checkpoint_path, model_dir, script_dir

import shutil
import sys
import os

model_name = sys.argv[1]

# subfoolder within model directory
save_dir= model_dir + model_name +"/"
if not os.path.exists(save_dir):
    print("model name does not exist")
else:
    print("loading model...")

config_fn = "config.py"
checkpoint_fn = "checkpoint.tar"

config_fp = script_dir+ config_fn


try:
    shutil.copyfile((save_dir + config_fn), config_fp )
    shutil.copyfile((save_dir + checkpoint_fn),checkpoint_path )
    print("model successfully loaded")
except:
    print("error on load")
 
