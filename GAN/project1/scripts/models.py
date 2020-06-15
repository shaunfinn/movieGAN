from config import image_size, ngpu, ngf, ndf, nc, nz

import torch
import torch.nn as nn
import torch.nn.parallel
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, ngpu):
      super(Generator, self).__init__()
      self.ngpu = ngpu
      self.numLayers = int(math.log2(image_size))-1   
      self.modLst = []    #module list
      self.ch_in = nz
      self.ch_out = int(2**(self.numLayers-2) * ngf)
      self.numch = nc
      for i in range(self.numLayers):
        if i ==0:
          self.modLst.append(nn.ConvTranspose2d( self.ch_in, self.ch_out, 4, 1, 0, bias=False))
          self.modLst.append(nn.BatchNorm2d(self.ch_out))
          self.modLst.append(nn.ReLU(True))
          
        elif (i == self.numLayers-1): # lastlayer
          self.ch_out = self.numch # number of image channels
          self.modLst.append(nn.ConvTranspose2d( self.ch_in, self.ch_out, 4, 2, 1, bias=False))
          self.modLst.append(nn.Tanh())

        else:
          self.modLst.append(nn.ConvTranspose2d( self.ch_in, self.ch_out, 4, 2, 1, bias=False))
          self.modLst.append(nn.BatchNorm2d(self.ch_out))
          self.modLst.append(nn.ReLU(True))
        
        #print("ch_in ",self.ch_in, "ch_out", self.ch_out )
        self.ch_in = self.ch_out
        self.ch_out =int(self.ch_out/ 2)

        self.main =  nn.Sequential(*self.modLst) 

    def forward(self, input):
        return self.main(input)

#Discriminator code
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.numLayers = int(math.log2(image_size))-1   
        self.modLst = []    #module list
        self.ch_in = nc
        self.ch_out = int(ndf)

        for i in range(self.numLayers):
          if i ==0:
            
            self.modLst.append(nn.Conv2d(self.ch_in, self.ch_out, 4, stride=2, padding=1, bias=False))
            self.modLst.append(nn.LeakyReLU(0.2, inplace=True))
          
            
          elif (i == self.numLayers-1): # lastlayer
            self.ch_out = 1 # real/fake value
            self.modLst.append(nn.Conv2d(self.ch_in, self.ch_out, 4, stride=1, padding=0, bias=False))
            self.modLst.append(nn.Sigmoid())
            
          else:
            self.modLst.append(nn.Conv2d(self.ch_in, self.ch_out, 4, stride=2, padding=1, bias=False))
            self.modLst.append(nn.BatchNorm2d(self.ch_out))
            self.modLst.append(nn.LeakyReLU(0.2, inplace=True))
          
          #print("ch_in ",self.ch_in, "ch_out", self.ch_out )
          self.ch_in = self.ch_out
          self.ch_out =int(self.ch_out* 2)

          self.main =  nn.Sequential(*self.modLst) 

    def forward(self, input):
        return self.main(input)
    
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Create the generator and discriminatpr
netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)

# DEBUG Print the model
#print(netG)
#print(netD)

