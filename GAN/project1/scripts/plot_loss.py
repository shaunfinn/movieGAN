import torch
import matplotlib.pyplot as plt


from config import checkpoint_path

checkpoint = torch.load(checkpoint_path)

G_losses = checkpoint['G_losses']
D_losses = checkpoint['D_losses']

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show() 
