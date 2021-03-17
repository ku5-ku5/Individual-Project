
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable

import PIL
from PIL import Image
from net import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ["negative_data", "without_mask"]

def Precision(true_pos, false_neg):
	return(true_pos / (true_pos + false_neg))


def Recall(true_pos, false_pos):
	return(true_pos / (true_pos + false_pos))

def F1(true_pos, false_pos, false_neg):
	p = Precision(true_pos, false_neg)
	r = Recall(true_pos, false_pos)
	return(2 * ((p * r) / (p + r)))

def confusion_matrix(net, testloader):
	net.eval()

	#initialise a 2x2 matrix to store the values in
	matrix = np.zeros((2,2),dtype=int)
	#keys = ["TP", "FP", "FN", "TN"]

	#matrix = dict.fromkeys(keys, 0)

	with torch.no_grad():
	    for data in testloader:
	        images, labels = data[0].to(device), data[1].to(device)
	        outputs = net(images)
	        _, predicted = torch.max(outputs.data, 1)

	        for i in range(images.size()[0]):
	        	if predicted[i].item() == 1 and labels[i].item() == 1:
	        		#Identify true positive
	        		matrix[0][0] += 1
	        	elif predicted[i].item() == 1 and labels[i].item() == 0: 
	        		#identify false positive 
	        		matrix[0][1] += 1
	        	elif predicted[i].item() == 0 and labels[i].item() == 1:
	        		#identify false negative
	        		matrix[1][0] += 1
	        	else:
	        		#identify true negative 
	        		matrix[1][1] += 1
	return(matrix)



#net = Net()
#net.load_state_dict(torch.load('masked_model.pth'))





