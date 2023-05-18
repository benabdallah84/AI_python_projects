import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
input_size  = 28#?column as features
sequense_lenght = 28#?rows as sequences
num_layers=2#?hidden layers
hidden_size = 256#?nodes in hidden layer
num_classes = 10
learning_rate = 0.001
num_epochs = 2
batch_size = 64


#create fully connected network
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        #! forward prop
        out,_=self.blstm(x,(h0,c0))
        
        out = self.fc(out[:,-1,:])
        return out

     
    

#load dataset
train_dataset = datasets.MNIST(root='datasets/',train= True,transform = transforms.ToTensor(), download =True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='datasets/',train= False,transform = transforms.ToTensor(), download =True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#initialize network
model = BLSTM(input_size, hidden_size,num_layers, num_classes).to(device)

#define loss function
criterion = nn.CrossEntropyLoss()

#define optimizer
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

#train network
for epoch in range(num_epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        #get data to cuda if possible
        data = data.to(device=device).squeeze(1)
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
            data = data.to(device=device).sequeeze(1)


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
