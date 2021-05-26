import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.optim as optim
from torch import topk
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
import skimage.transform



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ["negative_data", "with_mask"]




def retrieveImages(data_dir):


        image_transforms = transforms.Compose(
                           [transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        dataset = ImageFolder(
                              root = data_dir,
                              transform = image_transforms
                               )

        dataset.class_idx = {}

        for i in range(0, len(classes)):
                dataset.class_idx[classes[i]] = i

        eval_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, num_workers=2)


        return(eval_loader)



def return_CAM(feature_conv, weight, class_idx):
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        cam = weight.dot(feature_conv.reshape((nc, h*w)))

        # reshape CAM to match dimensions of activation images
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cv2.resize(cam_img, size_upsample)
        return [cam_img]


def create_fig(img_activations, preds_gt, title, batch_size=5):
    img_batch = [img_activations[i:i + batch_size] for i in range(0, len(img_activations), batch_size)]
    index = 0
    for img_list in img_batch:
        fig, plots = plt.subplots(6, len(img_list), figsize=(10, 10))
        fig.suptitle(title)
        for i in range(0, len(img_list)):
            if preds_gt["Predicted"][((index * batch_size) + i)] == preds_gt["GroundT"][((index * batch_size) + i)]:
                colour = "green"
            else:
                colour = "red"
            for j in range(6):
                plots[j][i].imshow(img_list[i][j].cpu())
                plots[j][i].set_title(preds_gt["Predicted"][((index * batch_size) + i)], color=colour)
        fig.tight_layout()
        plt.show()
        index += 1
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        fig.savefig("./figures/" + title + "_" + dt_string + ".png")


def run_CAM(net, evalloader, weight, title):
    net.eval()
    img_activations = []
    predictions = []
    ground_truths = []
    preds_gt = {"Predicted": [], "GroundT": []}

    for i, images in enumerate(evalloader, 0):
        image, label = images[0].to(device), images[1].to(device)
        img = net(image)
        probabilities = F.softmax(img, dim=1).data.squeeze()

        probs, idx = probabilities.sort(0, True)
        class_idx = topk(probs,1)[1].int()
        probs = probs.detach().cpu().numpy()
        idx = idx.cpu().numpy()

        pred_labels = idx[0]
        predicted = evalloader.dataset.classes[idx[0]]

        ground_truth = evalloader.dataset.classes[label]

        print("Predicted: " + predicted + ", Ground Truth: " + ground_truth)
        predictions.append(predicted)
        ground_truths.append(ground_truth)
        preds_gt["Predicted"].append(predicted)
        preds_gt["GroundT"].append(ground_truth)


        activation = {}

        # Used to retrieve indivudual activations
        def get_act(name):
            def act_hook(model, input, output):
                    activation[name] = output.detach()
            return act_hook

        net.conv1.register_forward_hook(get_act('conv1'))
        data, _ = image, label
        data.unsqueeze(0)
        output = net(data)
        act = activation['conv1'].squeeze()
        x = act[None,:, :, :]

        cam = return_CAM(x, weight, class_idx)
        t = transforms.Resize((256,256),interpolation=Image.NEAREST)
        img = image.squeeze()
        img_trans = t(img)
        plt.imshow(img_trans.permute(1, 2, 0))
        plt.imshow(skimage.transform.resize(cam[0], img_trans.shape[1:3]), alpha=0.5, cmap='jet');

        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        plt.savefig("./figures/" + title + "_" + dt_string + ".png")
        plt.show()

        trans = t(act)

        img_activations.append(trans)
    create_fig(img_activations, preds_gt, title)


if __name__ == '__main__':

    #Load model
    net = Net()
    net.to(device)

    net.load_state_dict(torch.load('masked_model.pth', map_location=torch.device(device)))

    weight_softmax_params = list(net._modules.get('conv1').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[-1].cpu().data.numpy())

    print(weight_softmax.shape)

    #Load images

    evalloader = retrieveImages('./data/evaluation')

    run_CAM(net, evalloader, weight_softmax, "Masked Model")

    #net.load_state_dict(torch.load('unmasked_model.pth', map_location=torch.device(device)))

    #run_CAM(net, evalloader, weight_softmax, "Unmasked Model")
