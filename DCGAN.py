import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_model
#! Things to try:
#? 1. What happenes if you use larger network?
#? 2. Better normalization with batchNorm
#? 3. Better learning rate
#? 4. Change architecture to a CNN



class Discriminator(nn.Module):
    def __init__(self,channels_img, features_d):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            # input: N * channels_img * 64 *64
            nn.Conv2d(channels_img, features_d, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),# 32*32
            self._block(features_d,features_d*2,4,2,1),#16*16
            self._block(features_d*2,features_d*4,4,2,1),#8*8
            self._block(features_d*4,features_d*8,4,2,1),#4*4
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),#1*1
            nn.Sigmoid()
        )
        
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self,x):
       return self.disc(x)
        
class Generator(nn.Module):
    def __init__(self,z_dim, channels_img, features_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            # input: N * z_dim * 1 * 1
            self._block(z_dim,features_g*16,4,1,0),#N*f_g*16 * 4 * 4
            self._block(features_g*16,features_g*8,4,2,1),#8*8
            self._block(features_g*8,features_g*4,4,2,1),#16*16
            self._block(features_g*4,features_g*2,4,2,1),#32*32
            nn.ConvTranspose2d(features_g*2,channels_img,kernel_size=4,stride=2,padding=1),
            nn.Tanh()#[-1,1]
            
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0,0.02)

def test():
    N, in_channels,H,W = 8,3,64,64   
    z_dim = 100
    x = torch.randn(N, in_channels, H, W)
    disc = Discriminator(in_channels,8)
    initialize_weights(disc)
    assert disc(x).shape == torch.Size([N, 1,1,1])
    gen = Generator(z_dim,in_channels,8)
    initialize_weights(gen)
    z = torch.randn(N, z_dim, 1, 1)
    assert gen(z).shape == torch.Size([N, in_channels, H, W])
    print('test passed')

#test()

        
#Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
lr = 2e-4
z_dim = 100
image_dim = 64
num_epoch = 1
batch_size = 128
channels_img = 1
features_disc=64
features_gen=64

my_transforms = transforms.Compose(
    [
        transforms.Resize((image_dim)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)]

        )
        
    ])



fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])#(0.1307,), (0.3081,)

dataset = datasets.MNIST(root='./datasets', download=True, transform=my_transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

disc = Discriminator(channels_img,features_disc).to(device)
gen = Generator(z_dim, channels_img, features_gen).to(device)

initialize_weights(disc)
initialize_weights(gen)

opt_disc = optim.Adam(disc.parameters(), lr=lr,betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise= torch.randn(32, z_dim,1,1).to(device)


writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

# steps to printing on tensorboard
step = 0

gen.train()
disc.train()

for epoch in range(num_epoch):
    for i, data in enumerate(loader):
        real_img, _ = data
        real_img = real_img.to(device)#.view(-1,784)
        #batch_size = real_img.shape[0]

        #train Discriminator: max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn((batch_size, z_dim,1,1)).to(device)
        fake_img = gen(noise)
        disc_real = disc(real_img).reshape(-1)

        disc_fake = disc(fake_img.detach())
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake_img).reshape(-1)
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (disc_loss_real + disc_loss_fake)/2
        opt_disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        opt_disc.step()

        #train Generator: min log(1- log(D(G(z)))
        output = disc(fake_img).reshape(-1)
        gen_loss = criterion(output, torch.ones_like(output))
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if i % 2 == 1:
            print(f"Epoch: {epoch}/{num_epoch}, Batch {i}/{len(loader)},Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")

            with torch.no_grad():
                fake= gen(fixed_noise)#.reshape(-1,1,28,28)
                #data=real_img.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_img[:32], normalize=True)
                             
                writer_fake.add_image("Mnist Fake Images", img_grid_fake,global_step=step)
                writer_real.add_image("Mnist Fake Images", img_grid_real,global_step=step)
            step +=1
    
    save_model(model=disc,target_dir=f"/content/checkpoints/discriminator/",model_name=f"discriminator_{epoch}.pth")
    save_model(model=gen,target_dir=f"/content/checkpoints/generator/",model_name=f"generator_{epoch}.pth")
            





                      
