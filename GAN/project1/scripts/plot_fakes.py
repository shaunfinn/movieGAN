import torch
import matplotlib.pyplot as plt
import numpy as np

import torchvision.utils as vutils
import sys

from config import nz, checkpoint_path
from models import netG

fig_size = int(sys.argv[1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


checkpoint = torch.load(checkpoint_path)
netG.load_state_dict(checkpoint['netG_state_dict'])


fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fake_batch = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(fig_size,fig_size))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake_batch.to(device), padding=0, normalize=True).cpu(),(1,2,0)))
plt.show()
    
