import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils.CatsAndDogsDataset import CatsAndDogsDataset

#load data
my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(200,200),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degree=45), 
    transforms.RandomVerticalFlip(p=0.05),   
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0])# value - mean / std

])
dataset = CatsAndDogsDataset(csv_file='Aumented_img/train.csv', root_dir='Aumented_img/dataset', transform=my_transform)

img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'img'+str(img_num) +'.png')
        img_num += 1

    
