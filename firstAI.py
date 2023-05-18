import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
'''
image_size  = 784 #28*28
num_classes = 10
mini_batch_size = 64
model = NN(image_size,num_classes)

x = torch.randn(mini_batch_size,image_size)

print(model(x))

'''
#set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyperparmeter
image_size  = 784 #28*28
num_classes = 10
learning_rate = 0.001
num_epochs = 1
batch_size = 64

#load dataset
train_dataset = datasets.MNIST(root='datasets/',train= True,transform = transforms.ToTensor(), download =True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='datasets/',train= False,transform = transforms.ToTensor(), download =True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#initialize network
model = NN(input_size=image_size,num_classes=num_classes).to(device)

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
        #set to correct shape
        data = data.reshape(data.shape[0],-1)

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
            data = data.reshape(data.shape[0],-1)
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