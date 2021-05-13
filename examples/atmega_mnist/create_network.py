#!/usr/bin/env python

# This creates a Convolutinal Neural Network using Pytorch to detect
# the hand-written characters from the MNIST dataset.
#
# The work is based on work by Arijit Mukherjee: https://www.youtube.com/watch?v=kI3F8lLNneM
# The original code does not mention a license. Since it is publicly available I assume it is
# fair to use like this. If not, let me know.
# The framework code is mostly unmodified, the network itself is changed to be 8-bit friendly

import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable as var


n_batch= 128
learning_rate = 0.0005
n_epoch = 20
n_print = 50
resize_to = 14
device = "cuda:0"

# A custom transform to binarize the input data.
# The demo device touch screen is not really sensitive enough
# to give more than binary touch data, so this makes the MNIST data
# look more like what the code on the AVR sees
class Binarize(object):
    def __call__(self, img):
        img[img < 0.3]=0
        img[img >= 0.3]=1
        return img
# Stack up all transforms performed on each input image. These augment the dataset nicely.
# Output is a 14x14x1 pixel image.
T = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.RandomRotation(15),
                                    torchvision.transforms.Resize(resize_to+2),
                                    torchvision.transforms.RandomCrop(resize_to),
                                    Binarize()
                                   ])

train_data = torchvision.datasets.MNIST('data',train=True,download=True,transform=T)
val_data = torchvision.datasets.MNIST('data',train=False,download=True,transform=T)

train_dl = torch.utils.data.DataLoader(train_data,batch_size = n_batch)
val_dl = torch.utils.data.DataLoader(val_data,batch_size = n_batch)


## If you want to have a look at the input
#import matplotlib.pyplot as plt
#dataiter = iter(train_dl)
#images, labels = dataiter.next()
#print(images.shape)
#print(labels.shape)
#plt.imshow(images[12].numpy().squeeze(), cmap='Greys_r')


#### Part II : Writing the Network
class myCNN(nn.Module):
  def __init__(self):
    super(myCNN,self).__init__()
    # input: 1 x 14 x 14
    # Conv2d( input_channels, output_channels, kernel_size)
    self.cnn1 = nn.Conv2d(1,20, 5, stride=2, bias=False)
    # 20 x 5 x 5
    self.cnn2 = nn.Conv2d(20,12,3, stride=1, bias=False)
    # 12 x 3 x 3

    self.linear = nn.Linear(12*3*3,10)
    self.relu = nn.ReLU()

  def forward(self,x):
    n = x.size(0)
    x = self.relu(self.cnn1(x))
    x = self.cnn2(x)
    x = x.view(n,-1)
    x = self.linear(x)
    return x


#### Part III : Writing the main Training loop

mycnn = myCNN().cuda()
cec = nn.CrossEntropyLoss()
optimizer = optim.Adam(mycnn.parameters(),lr = learning_rate)

def validate(model,data):
  # To get validation accuracy = (correct/total)*100.
  total = 0
  correct = 0
  for i,(images,labels) in enumerate(data):
    images = var(images.cuda())
    x = model(images)
    value,pred = torch.max(x,1)
    pred = pred.data.cpu()
    total += x.size(0)
    correct += torch.sum(pred == labels)
  return correct*100./total

for e in range(n_epoch):
  for i,(images,labels) in enumerate(train_dl):
    images = var(images.cuda())
    labels = var(labels.cuda())
    optimizer.zero_grad()
    pred = mycnn(images)
    loss = cec(pred,labels)
    loss.backward()
    optimizer.step()
    if (i+1) % n_print == 0:
      accuracy = float(validate(mycnn,val_dl))
      print('Epoch :',e+1,'Batch :',i+1,'Loss :',float(loss.data),'Accuracy :',accuracy,'%')


# Save the trained model as ONNX
input_names = [ "network_input" ]
output_names = [ "network_output" ]
dummy_input = torch.randn(1, 1, resize_to,  resize_to).cuda()
torch.onnx.export(mycnn, dummy_input, "mnist.onnx", input_names=input_names, output_names=output_names)

