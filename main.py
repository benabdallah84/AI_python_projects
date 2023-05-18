import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1.0, 2.0, 3.0, 4],[1,5,8,7]], dtype=torch.float32, device = device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

x= torch.empty(size=(3,3))
x= torch.zeros((3,3))
x=torch.rand((3,3))
x=torch.ones((3,3))
x= torch.eye(5,5)
x= torch.arange(start=0,end=5,step=1)
x= torch.linspace(start=0.1,end=5,steps=10)
x=torch.empty((1,5)).normal_(mean=0,std=1)
x=torch.empty((1,5)).uniform_(0,1)
x=torch.diag(torch.ones(3))
print(x)
tensor= torch.arange(4)
bool_tens =tensor.bool()
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_back = tensor.numpy()

x = torch.tensor([3,5,7])
y = torch.tensor([7,5,6])

z1= torch.empty(3)
torch.add(x,y, out=z1)
print(z1)
z1= torch.add(x,y)
z = x+y
print(z)
z = x-y
print(z)
z = torch.true_divide(x,y)
print(z)

t = torch.zeros(3)
t.add_(x)
t += x
print(t)

z = x.pow(2)
z = x ** 2
print(z)
z = x > 0
print(z)

x = torch.rand((5,5))
y= torch.rand((5,5))
#z= torch.empty(3)
z = torch.mm(x,y)

#wise multiplicaion: smane dim, side bu side
z1 = x * y
x = torch.tensor([3,5,7])
y = torch.tensor([7,5,6])

#dot multiplication: 1D
z2 = torch.dot(x,y)
#x1 = x.mm(y)
#x2 = x1.matrix_power(3)
print(f'matrice z:',z)
print(f'matrice z1:',z1)
print(f'matrice z2:',z2)

#batch multiplication
batch = 3 
n=2
m=2
p=2

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))

z = torch.bmm(tensor1,tensor2)
print(f'batch matrice',z)
print(f'dimentions: ',z.ndimension())
print(f'numel: ',z.numel())
# broadcasting
x1 = torch.tensor([[4,4,4],[3,3,3]])
x2 = torch.tensor([[2,2,2]])
z = x1 - x2
z2=x1 ** x2
print(z)
print(z2)
# other operations
x = torch.tensor([3,5,7])
y = torch.tensor([7,5,6])

sum_x =torch.sum(x,dim=0)
print(f'sum=',sum_x)
values, indices = torch.max(x,dim=0)
print(f'max=',values)
values, indices = torch.min(x, dim=0)
print(f'min=',values)
abs_x = torch.abs(x)
print(f'abs=',abs_x)
z = torch.argmax(x,dim=0)
print(f'argmax=',z)
z = torch.argmin(x,dim=0)
print(f'argmin=',z)
mean_x = torch.mean(x.float(),dim=0)
print(f'mean=',mean_x)
z=torch.eq(x,y)
print(f'eq=',z)
y_sorted, indices = torch.sort(y,dim=0,descending = False)
print(f'y_sorted:', y_sorted)

x =torch.tensor([-1,20,6,-0.4])
z  =torch.clamp(x , min=0,max=10)
print(f'clamp:',z)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z=torch.any(x)
print(f'any:',z)
z=torch.all(x)
print(f'all:',z)

#indexing 
batch_zize = 10
features = 25
x = torch.rand((batch_zize,features))
print(x[0]) #x[0,:]
print(x[:,0]) 
print(x[2,0:10])#9 features of the third element
x[0,0]=100

#fancy indexing
x= torch.arange(10)
indices = [2,5,3]
print(x[indices])

x= torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols])
#advanced
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])

print(x[x % 2 == 0]) # or x.remainder(2) ==0

#usefull operations
x = torch.tensor([[1,2,3,15,17,67]])
print(f'where condition: ',torch.where(x > 15 ,x , x*2))# where works with values
print(torch.tensor([0,0,1,1,5,2,3]).unique())
print(f'numel: ',x.numel())

# reshape stuffs
x= torch.arange(9)
x3x3= x.view(3,3)# faster that reschap
x_3x3=x.reshape(3,3)
y= x3x3.t()
print(y.contiguous().view(9)) # directly use reshape(9)
print(y.reshape(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim = 0).shape)
print(torch.cat((x1,x2), dim = 1).shape)

z1 = x1.view(-1)
z2 = x1.reshape(-1)

print(f'falatten with view',z1)
print(f'falatten with reshape',z1)
batch = 1
x = torch.rand((batch, 2, 5))
print(x)
z = x.view(batch, -1)
z = x.permute(0, 2,1) # pemute the matrix by permuting the index position
print(z)

x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x= torch.arange(10).unsqueeze(0).unsqueeze(1)# 1x1x10
z =x.squeeze(1)#1x10
print(z.shape)