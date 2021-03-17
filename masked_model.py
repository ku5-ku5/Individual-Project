import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, Subset, WeightedRandomSampler
import torchvision
from torch.autograd import Variable
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import os

import PIL
from PIL import Image
import warnings
from tqdm import tqdm
import time
import warnings
from net import *

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ["negative_data", "with_mask"]

def retrieveData():

	image_transforms = transforms.Compose(
	                   [transforms.Resize((32,32)),
	                    transforms.ToTensor(),
	                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


	data_dir = './data/training/mask_training'

	dataset = ImageFolder(
	                      root = data_dir,
	                      transform = image_transforms
	                       )

	dataset.class_idx = {}

	for i in range(0, len(classes)):
		dataset.class_idx[classes[i]] = i

	print(dataset.class_idx)

	# invert the class to index dictionary to create index to class dictionary
	idx_class = {v: k for k, v in dataset.class_idx.items()}

	def get_class_distribution(dataset_obj):
	    count = {}
	    for i in range(0, len(classes)):
	    	count[classes[i]] = 0

	    for element in dataset_obj:
	    	lbl_idx = element[1]
	    	lbl_idx = idx_class[lbl_idx]
	    	count[lbl_idx] += 1
	    return count
	print("Distribution of classes: \n", get_class_distribution(dataset))
	train_dataset, test_dataset = random_split(dataset, (5000, 1851))

	train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=4, num_workers=2)
	test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=4, num_workers=2)
	print("Length of the train_loader:", len(train_loader))
	print("Length of the test_loader:", len(test_loader))

	return(train_loader, test_loader)

def evaluate(net, trainloader):
	correct = 0
	total = 0
	with torch.no_grad():
	    for data in trainloader:
	        images, labels = data[0].to(device), data[1].to(device)
	        outputs = net(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	accuracy = 100 * (correct / total)
	return(accuracy)

def classAccuracies(net, trainloader):
	class_count = len(classes)
	class_correct = [0] * class_count
	class_total = [0] * class_count

	with torch.no_grad():
		for data in trainloader:
			images, labels = data[0].to(device), data[1].to(device)
			outs = net(images)
			_, predicted = torch.max(outs, 1)
			correct = (predicted == labels).squeeze()
			for i in range(3):
				label = labels[i]
				if correct[i].item() == True:
					class_correct[int(label.item())] += 1
				class_total[int(label.item())] += 1

	outtxt = ""
	for i in range(class_count):
		accuracy = 100 * (class_correct[i] / class_total[i])
		outtxt += 'Accuracy of ' + str(classes[i]) + ": " + str(accuracy) + "\n"
	return(outtxt)

def train(net, loss_fn, opt, trainloader, epochs):
    #num_steps = 0
    min_loss = 99999
    outtxt = ""

    for epoch in tqdm(range(1, epochs+1), total=epochs, desc='Training'):
    	time.sleep(0.1)
    	running_loss = 0.0
    	loss_list = []
    	net.train() # Setting the neural network to TRAIN mode
    	for i, data in enumerate(trainloader, 0):
    		inputs, labels = data[0].to(device), data[1].to(device) 

    		optimiser.zero_grad() 

    		outputs = net(inputs)
    		loss = criterion(outputs, labels)
    		loss.backward()
    		optimiser.step()
    		loss_list.append(loss.item()) 
    	loss = sum(loss_list) / len(loss_list)
    	acc = evaluate(net, trainloader) 

    	#Add accuracy and loss stats to output

    	outtxt += "\nEpoch: " + str(epoch) + "\nAccuracy: " + str(acc) + "\nloss: " + str(loss) + "\n"

    	#Add accuracy of individual classes to output

    	outtxt += classAccuracies(net, trainloader)

    	#Determine the best model based on loss 

    	if loss < min_loss:
    		min_loss = loss
    		bestmodel = net.state_dict()
    #torch.save(bestmodel,'masked_model.pth')
    print(outtxt)
    return None
	    

if __name__ == '__main__':
	#Load the image data in
	train_loader, test_loader = retrieveData()

	net = Net()
	net.to(device)

	#defining the loss function
	criterion = nn.CrossEntropyLoss()

	#defining the optimiser
	optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# epoch = a full iteration of training
	
	#epochs = input("Enter number of epochs: ")
	train(net, criterion, optimiser, train_loader, 2)
	print('Training Complete')

