
#BASIC SETTINGs:

folder_name = "project1"

image_size = 64   #size of images created in pixels ie 64x64 pixels. Values available: {8,16,32,64,128}

nc = 3            # Number of channels in the training images. For color images this is 3. For black and white use 1

nz = 100          #Size of z latent vector (generator input) 
                  #for image_size = 32,64,128 use 50,100,400 respectively

ngf = 64          # Size of feature maps in generator;
                  # set to image_size, unless image_size =128, then set to 64

ndf = 64          # Size of feature maps in discriminator;
                  # set to image_size. Unless image_size =128, then set to 32


#ADVANCED SETTINGS:


workers = 8         # Number of workers for dataloader

batch_size = 128    # Batch size during training

lr = 0.0002         # Learning rate for optimizers

beta1 = 0.5         # Beta1 hyperparam for Adam optimizers

ngpu = 1            # Number of GPUs available. Use 0 for CPU mode.

lsf = 0.2           # label softening factor  fake= 0:lsf; real = (1-lsf): 1


#DON'T EDIT! ...

root_dir = "/content/drive/My Drive/GAN/" + folder_name +"/"
dataset_dir = root_dir + "data/processed/"
rawdata_dir = root_dir + "data/raw/"
video_dir = root_dir + "video/"
model_dir = root_dir + 'model/'
script_dir = root_dir + 'scripts/'
checkpoint_path = root_dir + 'checkpoint/checkpoint.tar'
