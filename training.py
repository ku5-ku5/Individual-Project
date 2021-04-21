import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torch.autograd import Variable
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms
import os
import PIL
from PIL import Image
import warnings
from net import *
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def retrieveData(dataDir, classes, img_total):

	image_transforms = transforms.Compose(
	                   [transforms.Resize((32,32)),
	                    transforms.ToTensor(),
	                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


	dataset = ImageFolder(
	                      root = dataDir,
	                      transform = image_transforms
	                       )

	dataset.class_idx = {}

	for i in range(0, len(classes)):
		dataset.class_idx[classes[i]] = i

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

	train_split = round(0.85 * img_total)
	test_split = img_total - train_split

	train_dataset, test_dataset = random_split(dataset, (train_split, test_split))

	train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=4, num_workers=2)
	test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=4, num_workers=2)
	print("Length of the train_loader:", len(train_loader))
	print("Length of the test_loader:", len(test_loader))

	return(train_loader, train_loader)

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

def classAccuracies(net, trainloader, classes):
	class_count = len(classes)
	class_correct = [0] * class_count
	class_total = [0] * class_count

	with torch.no_grad():
		for data in trainloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			predictions = (predicted == labels).squeeze()
			for i in range(len(predictions)):
				label = labels[i]
				if predictions[i].item() == True:
					class_correct[int(label.item())] += 1
				class_total[int(label.item())] += 1

	outtxt = ""
	for i in range(class_count):
		accuracy = 100 * (class_correct[i] / class_total[i])
		outtxt += 'Accuracy of ' + str(classes[i]) + ": " + str(accuracy) + "\n"
	return(outtxt)

def train(net, lossFn, optimiser, classes, trainloader, epochs, filename=""):
    #num_steps = 0
    min_loss = 99999
    outtxt = ""
    accs = []
    losses = []

    savemodel = True
    if filename == "":
    	savemodel = False

    for epoch in range(1, epochs + 1):
    	running_loss = 0.0
    	loss_list = []

    	# Set the neural network to train mode
    	net.train()
    	for i, data in enumerate(trainloader, 0):
    		inputs, labels = data[0].to(device), data[1].to(device) 

    		#reset gradients to zero
    		optimiser.zero_grad() 

    		#forward propagation
    		outputs = net(inputs)
    		loss = lossFn(outputs, labels)

    		#backward propagation
    		loss.backward()

    		#optimise
    		optimiser.step()
    		loss_list.append(loss.item()) 
    	loss = sum(loss_list) / len(loss_list)
    	acc = evaluate(net, trainloader)

    	accs.append(acc)
    	losses.append(loss)
    	#Add accuracy and loss stats to output

    	outtxt += "\nEpoch: " + str(epoch) + "\nAccuracy: " + str(acc) + "\nloss: " + str(loss) + "\n"

    	#Add accuracy of individual classes to output

    	outtxt += classAccuracies(net, trainloader, classes)

    	#Determine the best model based on loss 

    	if loss < min_loss:
    		min_loss = loss
    		bestmodel = net.state_dict()
    print(outtxt)
    fig, a = plt.subplots()

    upper = len(losses) + 1
    x = [i for i in range(1, upper)]

    a.plot(x, losses, color="blue")
    a.set_xlabel("Training Iterations", fontsize=12)
    a.set_ylabel("Loss", fontsize=12, color="blue")
    b = a.twinx()
    b.plot(x, accs, color="green")
    b.set_ylabel("Accuracy (%)", fontsize=12, color="green")

    if savemodel == True:
    	torch.save(bestmodel, filename + '.pth')
    	fig.savefig(filename + '.png')

    plt.show()




    return None