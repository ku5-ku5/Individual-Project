import numpy as np
import pandas as pd
from datetime import datetime
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
import torch.nn.functional as nnf
import cv2



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ["negative_data", "with_mask"]




def retrieveImages():


        image_transforms = transforms.Compose(
                           [transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        data_dir = './data/evaluation'

        dataset = ImageFolder(
                              root = data_dir,
                              transform = image_transforms
                               )

        dataset.class_idx = {}

        for i in range(0, len(classes)):
                dataset.class_idx[classes[i]] = i

        eval_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, num_workers=2)


        return(eval_loader)


'''
def return_CAM(feature_conv, weight, class_idx):
        size_upsample = (256, 256)
        nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
                cam = weight[idx].dot(feature_conv.reshape((nc, h * w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
        return(cam_img)
'''


def run_CAM(net, evalloader, weight):
    net.eval()
    img_activations = []
    for i, images in enumerate(evalloader, 0):
        image, label = images[0].to(device), images[1].to(device)
        img = net(image)
        probabilities = F.softmax(img, dim=1).data.squeeze()

        probs, idx = probabilities.sort(0, True)
        probs = probs.detach().cpu().numpy()
        idx = idx.cpu().numpy()

        pred_labels = idx[0]
        predicted = evalloader.dataset.classes[idx[0]]
        #w = weight[:, pred_labels]
        #w1 = torch.from_numpy(w)
        #cam = img.dot(w1)

        ground_truth = evalloader.dataset.classes[label]

        print("Predicted: " + predicted + ", Ground Truth: " + ground_truth)

        activation = {}
        def get_act(name):
                def hook(model, input, output):
                        activation[name] = output.detach()
                return hook

        net.conv1.register_forward_hook(get_act('conv1'))
        data, _ = image, label
        data.unsqueeze(0)
        output = net(data)
        act = activation['conv1'].squeeze()
        #fig, plots = plt.subplots(act.size(0))
        #out = nnf.interpolate(act, size=32, mode='nearest')
        #print(out.shape)
        t = transforms.Resize((128,128),interpolation=Image.NEAREST)
        trans = t(act)
        img_activations.append(trans)

    fig, plots = plt.subplots(6, len(img_activations))
    for i in range(len(img_activations)):
            for j in range(6):
                plots[j][i].imshow(img_activations[i][j].cpu())
    plt.xticks([])
    plt.yticks([])
    plt.show()
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    fig.savefig("./figures/" + dt_string + ".png")
        #break
        #plt.show()
        #print(act.shape)
        #act = act.unsqueeze(1)
        #trans = transforms.Compose([transforms.Resize(32)])
        #tData = trans(act)
        #print(tData.shape)
        #return_CAM(tData, weight, idx[0])

if __name__ == '__main__':

        #Load model
    net = Net()
    net.to(device)

    net.load_state_dict(torch.load('masked_model.pth', map_location=torch.device(device)))

        #params = list(Net().parameters())
        #weight = np.squeeze(params[-1].data.numpy())

    weight_softmax_params = list(net._modules.get('conv1').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    #Load images

    evalloader = retrieveImages()

    run_CAM(net, evalloader, weight_softmax)

    #net.load_state_dict(torch.load('unmasked_model.pth', map_location=torch.device(device)))

    #run_CAM(net, evalloader, weight_softmax)
