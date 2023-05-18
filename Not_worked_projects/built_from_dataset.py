import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import DataLoader
from utils.CatsAndDogsDataset import CatsAndDogsDataset
from utils.save_load_functions import save_checkpoint,load_checkpoint
from utils.random_split import random_split
# def save_checkpoint(state, filename="checkpoint.pth.tar"):
#     print("=> Saving checkpoint..")
#     torch.save(state, filename)

# def load_checkpoint(checkpoint):
#     print("=> Loading checkpoint..")
#     checkpoint = torch.load('checkpoint.pth.tar')
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print(checkpoint['epoch'])
#     return checkpoint['epoch']
#Hyperparmeter
in_channels  = 3
num_classes = 10
learning_rate = 0.001
num_epochs = 1
init_epoch = 0
batch_size = 32
#load_model = True
#set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#create fully connected network

dataset = CatsAndDogsDataset(csv_file='datasets/testFolder/cats_dogs.csv',root_dir='datasets/testFolder/dataset', transform= transforms.ToTensor())
test_size = int(len(dataset)*0.2)
train_size = len(dataset)- int(len(dataset)*0.2)

print(len(dataset))
print(train_size + test_size)
train_dataset, test_dataset =torch.utils.data.random_split(dataset,[train_size,test_size])
# for img in train_dataset:
#     print(torch.tensor(img[0]).shape)
#print(torch.tensor(train_dataset[0][0]).shape)
#print(torch.tensor(test_dataset[0][0]).shape)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
#initialize network
model = torchvision.models.googlenet(pretrained = True)

#define loss function
criterion = nn.CrossEntropyLoss()

#define optimizer
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if os.path.exists('checkpoint.pth.tar'): 
   init_epoch = load_checkpoint(model, optimizer, 'checkpoint.pth.tar')
  
# batch_idx,(data,target) = enumerate(train_loader)
# print(batch_idx)
# print(data.shape)
# print(target.shape)

for ep in range(init_epoch,num_epochs):
    losses=[]
    
    
    for batch_idx, payload in enumerate(train_loader):
        #get data to cuda if possible
        data,target = payload
        
        data = data.to(device=device)
        target = target.to(device=device)
        
        #forward
        score = model(data)
        loss = criterion(score, target)
        losses.append(loss.item())
        #backward
        optimizer.zero_grad()
        #loss.requires_grad = True
        loss.backward()

        #grediant descend or adam step

        optimizer.step()
    mean_loss = sum(losses)/len(losses)
    print(f'Loss in epoch {ep} was {mean_loss:.3f}')
    if ep % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': ep}
        save_checkpoint(checkpoint)
        print('epoch',ep)   
#check accuracy on training and test test to see how good are our model
def check_accuraccy(loader,model):
    num_correct = 0
    num_samples=0
    model.eval()
    if loader.dataset.train:
        print("Checking accurracy on training data")
    else:
        print("Checking accurracy on test data")
    with torch.no_grad():
        for data,target in loader:
            data = data.to(device=device)
            target = target.to(device=device)
            
            score = model(data) 
            _, prediction = score.max(1)

            num_correct += (prediction == target).sum()#.item()
            num_samples += prediction.size(0)

        accuracy = (float(num_correct) / float(num_samples))*100
        print(f'Got accuracy: {accuracy:.2f}%') 

    model.train()
    #return accuracy

check_accuraccy(train_loader, model)
check_accuraccy(test_loader, model)