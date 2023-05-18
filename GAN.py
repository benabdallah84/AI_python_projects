import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

#! Things to try:
#? 1. What happenes if you use larger network?
#? 2. Better normalization with batchNorm
#? 3. Better learning rate
#? 4. Change architecture to a CNN



class Discriminator(nn.Module):
    def __init__(self,image_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim,128),#784=28*28*1
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x):
       return self.disc(x)
        
class Generator(nn.Module):
    def __init__(self,z_dim, image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,image_dim),
            nn.Tanh(),#!since the input was normalised to [-1,1], the output also need to be tanh

        )
    def forward(self,x):
        return self.gen(x)
        
#Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
lr = 3e-4
z_dim = 64 #128,256
image_dim = 28 * 28 * 1 #784
num_epoch = 50
batch_size = 32

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

fixed_noise = torch.randn(batch_size, z_dim).to(device)
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])#(0.1307,), (0.3081,)

dataset = datasets.MNIST(root='./datasets', download=True, transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

gen.train()
disc.train()
step = 0
for epoch in range(num_epoch):
    for i, data in enumerate(loader):
        real_img, _ = data
        real_img = real_img.view(-1,784).to(device)
        batch_size = real_img.shape[0]

        #train Discriminator: max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake_img = gen(noise)
        disc_real = disc(real_img).view(-1)

        #disc_fake = disc(fake_img.detach())
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake_img).view(-1)
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (disc_loss_real + disc_loss_fake)/2
        opt_disc.zero_grad()
        disc_loss.backward(retain_graph=True)

        opt_disc.step()

        #train Generator: min log(1- log(D(G(z)))
        output = disc(fake_img).view(-1)
        gen_loss = criterion(output, torch.ones_like(output))
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if i == 0:
            print(f"Epoch: {epoch}/{num_epoch}, Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")

            with torch.no_grad():
                fake= gen(fixed_noise).reshape(-1,1,28,28)
                data=real_img.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                             
                writer_fake.add_image("Mnist Fake Images", img_grid_fake,global_step=step)
                writer_real.add_image("Mnist Fake Images", img_grid_real,global_step=step)
                step +=1

            





                      
