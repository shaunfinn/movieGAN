from config import video_dir, checkpoint_path, image_size


import numpy as np
import math
import sys


import cv2
import glob
import shutil 
import os

from ISR.models import RRDN

rrdn = RRDN(weights='gans')
#rdn = RDN(weights='psnr-small')

movie_name = sys.argv[1]
fps =  int(sys.argv[2])      # frames per second
smoothing =  float(sys.argv[3])  # number of seconds between images
duration= int(sys.argv[4])


# video name
videotype = ".mp4" 
video_name = movie_name + "_ISR"
video_path = video_dir + video_name + "_sm" + str(smoothing) +"_dur" + str(duration) +"_fps" +str(fps)+ videotype
#print(video_path)

#create temporary folder to store images
temp_dir = "temp_interpImgs"


#remove/make directory where images are stored  /contents/temp_interp
#print(os.getcwd())
if os.path.exists(temp_dir):
    print("exists")
else:
    print("directory empty, make movie in low resolution first")

#covert saved images to movie
path_str = temp_dir+'/*.jpg'
img_array = []

pathlist=glob.glob(path_str)
pathlist.sort()
#print(pathlist)

#output size from ISR
size = (image_size*4,image_size*4)
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for filename in pathlist:
    img = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_sr = rrdn.predict(img_rgb)
    out.write(img_sr)
    
    
out.release() 
print("movie done")


 
#remove temp directory
#if os.path.exists(temp_dir):
#    shutil.rmtree(temp_dir)
