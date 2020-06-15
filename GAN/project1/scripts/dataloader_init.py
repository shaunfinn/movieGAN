from config import image_size, batch_size, ngpu, workers, dataset_dir

import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#transforms for dataloader: uncomment as required
#radnom crop and flip = data augmentation 
transform=transforms.Compose([
                                #transforms.RandomCrop(image_size, padding=int(image_size*0.1)),
                                
                                #transforms.RandomHorizontalFlip(),

                                transforms.Resize(image_size),  #scales image so smallest dimension= image_size
                                
                                transforms.CenterCrop(image_size),
                                
                                transforms.ToTensor(),
                                
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
     
                                ])

#Dataset from imagefolder
dataset = dset.ImageFolder(root=dataset_dir, transform=transform)

#Datatset for cifar10 
#dataset = dset.CIFAR10(root="/content/drive/My Drive/data", train=True, transform=transform, download=True)
  
# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers, drop_last=True)

if __name__ == "__main__":
    # Plot some training images
    real_batch = next(iter(dataloader))
    #print(torch.min(real_batch[0]), torch.max(real_batch[0]))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    #normalise flag here sets it between [0,1]
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=0, normalize=True).cpu(),(1,2,0)))
    plt.show()
