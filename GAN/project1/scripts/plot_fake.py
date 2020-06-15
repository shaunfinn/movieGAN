import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys

from config import nz, checkpoint_path
from models import netG

fig_size = int(sys.argv[1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


checkpoint = torch.load(checkpoint_path)
netG.load_state_dict(checkpoint['netG_state_dict'])

 
#normalise image to [0,1] - previously [-0.5,0.5]
inv_normalize = transforms.Compose([                             
    transforms.Normalize(mean=[-1.0,-1.0,-1.0], std=[2.0,2.0,2.0])
])

fixed_noise = torch.randn(1, nz, 1, 1, device=device)
fake = netG(fixed_noise).detach().cpu()
fake= inv_normalize(fake[0]) # take first from batch
#check normalisation
#print("min:  ", torch.min(fake), " max: ", torch.max(fake))
plt.figure(figsize=(fig_size,fig_size))
#plt.subplot()
plt.axis("off")
plt.title("Fake Image")
plt.imshow(np.transpose(fake,(1,2,0)))
plt.show()

