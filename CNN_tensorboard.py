import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
'''
model = CNN()
x=torch.randn(64,1,28,28)
print(model(x).shape)
exit
'''


#set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyperparmeter
in_channels  = 1
num_classes = 10
#learning_rate = 0.001
num_epochs = 1
#batch_size = 64

#load dataset
train_dataset = datasets.MNIST(root='datasets/',train= True,transform = transforms.ToTensor(), download =True)

test_dataset = datasets.MNIST(root='datasets/',train= False,transform = transforms.ToTensor(), download =True)




batch_sizes= [256] #[2,64]
learning_rates= [0.001]#[0.01,0.001,0.0001]
classes = ['0','1','2','3','4','5','6','7','8','9']

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        #train network
        step = 0
        
        #initialize network
        model = CNN(in_channels=in_channels,num_classes=num_classes)
        model.to(device)
        #define loss function
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        #define optimizer
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}')#tensorboard --logdir runs
        for epoch in range(num_epochs):
            losses = []
            accuracies = []

            for batch_idx,(data,target) in enumerate(train_loader):
                #get data to cuda if possible
                data = data.to(device=device)
                target = target.to(device=device)
                
                
                #forward
                score = model(data)
                loss = criterion(score, target)
                losses.append(loss)

                #backward
                optimizer.zero_grad()
                #loss.requires_grad = True
                loss.backward()

                #grediant descend or adam step

                optimizer.step()

                 
                #calculate 'running' training accuracy
                features = data.reshape(data.shape[0],-1)
                img_grid = torchvision.utils.make_grid(data)
                _, prediction = score.max(1)
                correct = prediction.eq(target).sum()
                accuracy = (float(correct) / float(data.size(0)))
                accuracies.append(accuracy)
                #! Do some transfroms
                
                class_lables = [classes[label] for label in prediction]
                writer.add_image('transformed_images',img_grid)
                writer.add_scalar('Training loss',loss,global_step=step)
                writer.add_scalar('Training accuracy',accuracy,global_step=step)
                if batch_idx == 230:
                    writer.add_embedding(features,metadata=class_lables,label_img=data, global_step=batch_idx)
                #! Adding histogram of weights
                #writer.add_histogram('fc1',model.fc1.weight, global_step=batch_idx)

                step += 1
            
            
            
            #! Addiing hparameters
            writer.add_hparams({'learning_rate':learning_rate,'batch_size':batch_size},
                               {'Accuraccy':sum(accuracies)/len(accuracies)}
                               )

    
            print(f'=====Results for batch size {batch_size} and learning rate {learning_rate}=====')

            print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%')

#check accuracy on training and test test to see how good are our model

def check_accuraccy(dataset,model):
    for batch_idx in batch_sizes:
        loader = DataLoader(dataset=dataset,batch_size=batch_idx,shuffle=True)
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
            print(f'Got test accuracy for {batch_idx}: {accuracy:.2f}%') 

        model.train()
        #return accuracy

check_accuraccy(train_dataset, model)
check_accuraccy(test_dataset, model)