# copies current checkpoint.tar and config.py files to  model_dir

from config import checkpoint_path, model_dir, script_dir

import shutil
import sys
import os

model_name = sys.argv[1]

# subfoolder within model directory
save_dir= model_dir + model_name +"/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print("created model subfolder")
else:
    print("model name already exists; overwriting...")

config_fn = "config.py"
checkpoint_fn = "checkpoint.tar"

config_fp = script_dir+ config_fn


try:
    shutil.copyfile(config_fp, (save_dir + config_fn))
    shutil.copyfile(checkpoint_path , (save_dir + checkpoint_fn))
    print("model successfully saved")
except:
    print("error on save")
