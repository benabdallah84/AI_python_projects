import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Hyperparmeter
in_channels  = 1
num_classes = 10
learning_rate = 0.001
num_epochs = 24
init_epoch = 0
batch_size = 64
#load_model = True
#set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#create fully connected network
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same convolution
        self.fc1 = nn.Linear(16*7*7,num_classes)#since we have 2 paxpooling layers 28/2=7 28/2=7-->7*7*16(num_channels)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x = self.pool(x)
        x=F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x=self.fc1(x)
        return x
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint..")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint..")
    checkpoint = torch.load('checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(checkpoint['epoch'])
    return checkpoint['epoch']
    
# def load_checkpoint(checkpoint, model, optimizer = None):

#     if not os.path.exists(checkpoint):
#         raise("File does not exists {}".format(checkpoint))
    
#     print("=> Loading checkpoint..")
#     checkpoint = torch.load(checkpoint)
#     model.load_state_dict(checkpoint['state_dict'])


#     if optimizer:
#         optimizer.load_state_dict(checkpoint['optimizer'])



#load dataset
train_dataset = datasets.MNIST(root='datasets/',train= True,transform = transforms.ToTensor(), download =True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='datasets/',train= False,transform = transforms.ToTensor(), download =True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#initialize network
model = CNN().to(device)

#define loss function
criterion = nn.CrossEntropyLoss()

#define optimizer
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if os.path.exists('checkpoint.pth.tar'): 
   init_epoch = load_checkpoint('checkpoint.pth.tar')
  
   
#train network
for ep in range(init_epoch,num_epochs):
    losses=[]
    
    
    for batch_idx,(data,target) in enumerate(train_loader):
        #get data to cuda if possible
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