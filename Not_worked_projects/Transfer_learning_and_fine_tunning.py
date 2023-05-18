import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        x = x.reshape(x.shape[0],-1)
        return x
    
#set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyperparmeter
in_channels  = 1
num_classes = 10
learning_rate = 0.001
num_epochs = 5
batch_size = 64

#load dataset
train_dataset = datasets.MNIST(root='datasets/',train= True,transform = transforms.ToTensor(), download =True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='datasets/',train= False,transform = transforms.ToTensor(), download =True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#load VGG16 model and modified it 
model = torchvision.models.vgg16(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier[0] = nn.Linear(512,10)
model.classifier[3] = nn.Linear(64,10)
model.classifier[6] = nn.Linear(64,10)
model.to(device)

#define loss function
criterion = nn.CrossEntropyLoss()

#define optimizer
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

#train network
for epoch in range(num_epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        #get data to cuda if possible
        data = data.to(device=device)
        target = target.to(device=device)
        
        
        #forward
        score = model(data)
        loss = criterion(score, target)

        #backward
        optimizer.zero_grad()
        #loss.requires_grad = True
        loss.backward()

        #grediant descend or adam step

        optimizer.step()
        
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