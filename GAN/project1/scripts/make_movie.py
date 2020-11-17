from models import netG
from config import nz, video_dir, checkpoint_path


import torch
import torchvision.utils as vutils
import numpy as np
import math
import sys

from scipy import interpolate
import cv2
import glob
import shutil 
import os

movie_name = sys.argv[1]
fps =  int(sys.argv[2])      # frames per second
smoothing =  float(sys.argv[3])  # number of seconds between images
duration= int(sys.argv[4])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#INTERPOLATION & MAKE MOVIE IMAGES 
#interps between generations (from random) imgs within the latent space and decodes to create inbetween images


#VIDEO SETTINGS
points = int(smoothing*fps)   # SMOOTHING, interpolation points (images) between each random image sample/ GRANULAITY/ Smoothing
samples = int(duration/smoothing)   #number sample images in movie. total images= points* noOf imgs
                    # set to POINTS if want all
interpType = 'cubic'
frames = samples * points 
# video name
videotype = ".mp4" 
video_name = movie_name
video_path = video_dir + video_name + "_sm" + str(smoothing) +"_dur" + str(duration) +"_fps" +str(fps)+ videotype
#print(video_path)

#create temporary folder to store images
temp_dir = "temp_interpImgs"
netG.eval()

#remove/make directory where images are stored  /contents/temp_interp
#print(os.getcwd())
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
else:
    os.mkdir(temp_dir)

#total number of images to be in the movie to base10- for zfill of image name  
imgs_log10 = math.ceil(math.log10(frames)) 

#get noise samples
# need an extra sample one for start/end
noise_samples  = np.random.randn(samples+1, nz, 1, 1)
points_lst = list( map(int, range(0,frames+1, points)))
#print(points_lst)
linfit = interpolate.interp1d(points_lst, noise_samples, kind='cubic', axis=0)
#print(linfit(1).shape)

#makes images and stores in temp directory
with torch.no_grad():
    x= linfit(0)
    #creates array length [frames * nz]
    for i in range(1,frames):
      x= np.append(x,linfit(i))
    #print(x.shape)
    x= x.reshape(-1,nz,1,1)
    #print(x.shape)
    z = torch.FloatTensor(x)
    z = z.to(device)
    genImgs = netG(z).detach().cpu()
    #print(genImgs.size())
    
    for i in range(genImgs.size(0)):
      img_fp = temp_dir+"/"+str(i).zfill(imgs_log10)+ ".jpg"
      vutils.save_image(genImgs[i, :, :, :],img_fp, normalize=True )
      #print(a)
      #plt.imshow(img.permute(1, 2, 0))
      #plt.savefig(img_name)

#covert saved images to movie
path_str = temp_dir+'/*.jpg'
img_array = []

pathlist=glob.glob(path_str)
pathlist.sort()
#print(pathlist)

for filename in pathlist:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release() 
print("movie done")
 
#remove temp directory
#if os.path.exists(temp_dir):
#    shutil.rmtree(temp_dir)
