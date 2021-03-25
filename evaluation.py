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


net = Net()
net.load_state_dict(torch.load('masked_model.pth'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


net.to(device)


for child in net.children():
	print(child)


#Load model


#Load image


#CAM function


#show output image with heatmap

'''

params = list(Net().parameters())
weight = np.squeeze(params[-1].data.numpy())

def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))
        cam = np.matmul(weight[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


'''

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


for fname in IMG_URL:
    
    fname = fname.rstrip('\n')    
    img_pil = Image.open(org_loc+fname+'.png')
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = model(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
 
    probs, idx = h_x.sort(0, True)
    probs = probs.detach().numpy()
    idx = idx.numpy()
    
    predicted_labels.append(idx[0])
    predicted =  train_loader.dataset.classes[idx[0]]
    
    print("Target: " + fname + " | Predicted: " +  predicted) 
 
    features_blobs = mod(img_variable)
    features_blobs1 = features_blobs.cpu().detach().numpy()
    CAMs = return_CAM(features_blobs1, weight, [idx[0]])

    readImg = org_loc+fname+'.png'
    img = cv2.imread(readImg)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img * 0.5
  
    cv2.imwrite("image_1", result)

