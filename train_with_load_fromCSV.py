import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os 
import numpy as np
import argparse

def getData(filname):
	# images are 48x48 = 2304 size vectors
	# N = 35887
	Y = []
	X = []
	first = True
	for line in open(filname):
		if first:
			first = False
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(p) for p in row[1].split()])
	X, Y = np.array(X) / 255.0, np.array(Y) # scaling is already done here
	X=X.reshape(35887,1,48,48)
	return X,Y

def get_loader(X,Y,args):
    tensor_X = torch.stack([torch.from_numpy(np.array(i)) for i in X])
    tensor_y = torch.stack([torch.from_numpy(np.array(i)) for i in Y])

    train_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size)
    return train_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #44x(48-5+1)(48-5+1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 24, 5)
        self.fc1 = nn.Linear(24 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
                num_features *=s
        return num_features
# construct model and set loss and optimizer
#create train function

def train_fn(train_loader,net, optimizer,criterion, epochs):   
    for e in range(epochs):
        running_loss =0
        correct = 0
        total = 0
        collect_normalizer=0
        for i, data in enumerate(train_loader):
            #print(i, len(data), data[0]['data'].size(), data[0]['label'].size())
            #print(i, data[0]['data'].permute(0, 3, 1, 2).size(), data[0]['label'].size())
            inputs = data[0]
            labels = data[1].squeeze().type(torch.LongTensor)
            #print(inputs.size(), labels.size())
            # zero the parameter gradients
            collect_normalizer += len(data[0])
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            pred,predicted =torch.max(outputs.data,1)
            pred_list=predicted.data.tolist()
            correct += (predicted == labels).sum().item()
            #acc=round(100 * correct / len(data[0]["data"]),4)
            #print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            #print("loss is",loss.item())
            running_loss += loss.item()
            #print(collect_normalizer)
        print('Epoch [{}/{}],  ACC {:.4f}'.format(e + 1, epochs, correct / collect_normalizer ))

    return net, optimizer, train_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./', type=str, 
                        help='where is the csv file')
    parser.add_argument('--cpu', default='Yes', type=str, 
                        help='using GPU or CPU')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch_size , default to 512')
    parser.add_argument('--device', default=0, type=int,
                        help='which gpu to use')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    # get data using pytorch default dataLoader
    filname=args.data+'fer2013.csv'
    X,Y=getData(filname)
    train_loader= get_loader(X,Y,args)    
    ## the model 
        
    if args.cpu=='No' :
        which_device=str(args.device)
        device = torch.device("cuda:{}".format(which_device) if torch.cuda.is_available() else "cpu") # train on gpu number 0 
        print("using GPU device ", device )
    else:
        print("using CPU only ")
    # construct the model and move to GPU if using GPU to train
    net=Net()
    if not args.cpu :
        torch.cuda.set_device(device)
        net.cuda()
        criterion = nn.CrossEntropyLoss().cuda(device)
    else:
        criterion = nn.CrossEntropyLoss()    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # start the training loop
    epochs=int(args.epochs)
    net, optimizer, train_loader =train_fn(train_loader, net, optimizer,criterion, epochs)

if __name__ == '__main__':    
    classes = ['Anger', 'Disgust', 'Fear', 'Happy','Sad','Surprise', 'Neutral']
    num_class=len(classes)
    main()
    print("training complete !")



