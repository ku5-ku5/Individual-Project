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
from  torch.nn.modules.upsampling import Upsample



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ["negative_data", "without_mask"]




def retrieveImages():


	image_transforms = transforms.Compose(
	                   [transforms.Resize((32,32)),
	                    transforms.ToTensor(),
	                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


	data_dir = './data/training/maskless_training'

	dataset = ImageFolder(
	                      root = data_dir,
	                      transform = image_transforms
	                       )

	dataset.class_idx = {}

	for i in range(0, len(classes)):
		dataset.class_idx[classes[i]] = i

	print(dataset.class_idx)

	eval_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, num_workers=2)


	return(eval_loader)




params = list(Net().parameters())
weight = np.squeeze(params[-1].data.numpy())


def run_CAM(net, evalloader, weight):
    net.eval()
    for i, images in enumerate(evalloader, 0):
    	image, label = images[0].to(device), images[1].to(device)
    	img = net(image)
    	h_x = F.softmax(img, dim=1).data.squeeze()

    	probs, idx = h_x.sort(0, True)
    	probs = probs.detach().cpu().numpy()
    	idx = idx.cpu().numpy()

    	pred_labels = idx[0]
    	predicted = evalloader.dataset.classes[idx[0]]

    	ground_truth = evalloader.dataset.classes[label]

    	activation = {}

    	def get_activation(name):
    		def hook(model, input, output):
    			activation[name] = output.detach()
    		return hook

    	net.conv1.register_forward_hook(get_activation('conv1'))
    	data, _ = image, label
    	output = net(data)
    	act = activation['conv1'].squeeze()

    	fig, axarr = plt.subplots(act.size(0))
    	
    	for idx in range(act.size(0)):
    		axarr[idx].imshow(act[idx].cpu())
    	
    	plt.show()

if __name__ == '__main__':

	#Load model

	net = Net()
	net.to(device)
	net.load_state_dict(torch.load('masked_model.pth'))

	params = list(Net().parameters())
	weight = np.squeeze(params[-1].data.numpy())

	#Load images

	evalloader = retrieveImages()



	run_CAM(net, evalloader, weight)

	#CAM function


	#show output image with heatmap