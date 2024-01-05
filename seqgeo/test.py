import argparse
import os
import numpy as np
import math
import itertools

from numpy.lib.function_base import average

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import SatelliteFeatureExtractor, StreetFeatureExtractor, TransMixer
from dataset import ImageDataset


STREET_IMG_WIDTH = 320
STREET_IMG_HEIGHT = 180
SATELLITE_IMG_WIDTH = 256
SATELLITE_IMG_HEIGHT = 256
SEQUENCE_SIZE = 7

def ValidateOne(distArray, topK):
    acc = 0.0
    dataAmount = 0.0
    for i in range(distArray.shape[0]):
        groundTruths = distArray[i,i]
        pred = torch.sum(distArray[:,i] < groundTruths)
        if pred < topK:
            acc += 1.0

        dataAmount += 1.0
    return acc / dataAmount

def ValidateAll(streetFeatures, satelliteFeatures):
    distArray = 2 - 2 * torch.matmul(satelliteFeatures, torch.transpose(streetFeatures, 0, 1))
    topOnePercent = int(distArray.shape[0] * 0.01) + 1
    valAcc = torch.zeros((1, topOnePercent))
    for i in range(topOnePercent):
        valAcc[0,i] = ValidateOne(distArray, i)

    return valAcc

def InferOnce(grdFE, satFE, transMixer, batch, device, noMask):
    grdImgs = batch["street"].to(device)
    sateImgs = batch["satellite"].to(device)

    numSeqInBatch = grdImgs.shape[0]

    #street view featuer extraction
    grdImgs = grdImgs.view(grdImgs.shape[0]*grdImgs.shape[1],\
        grdImgs.shape[2],grdImgs.shape[3], grdImgs.shape[4])

    grdFeature = grdFE(grdImgs)
    grdFeature = grdFeature.view(numSeqInBatch, SEQUENCE_SIZE, -1)

    #satellite view feature extraction
    sateImgs = sateImgs.view(sateImgs.shape[0], sateImgs.shape[1]*sateImgs.shape[2],\
        sateImgs.shape[3], sateImgs.shape[4])
    sateFeature = satFE(sateImgs)
    sateFeature = sateFeature.view(numSeqInBatch, -1)
    # print(sateFeature.shape)

    if not noMask:
        grdMixedFeature = transMixer(grdFeature, mask=True, masked_range = [0,6], max_masked=opt.num_grd_image)
    else:
        grdMixedFeature = transMixer(grdFeature, mask=False, masked_range = [0,6])
    grdGlobalFeature = grdMixedFeature.permute(0,2,1)
    grdGlobalLatent = F.avg_pool1d(grdGlobalFeature, grdGlobalFeature.shape[2]).squeeze(2)


    return sateFeature, grdGlobalLatent

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=50, help="which epoch model to load")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--model_folder", type=str, default='SAVE_NAME', help='name of the model')
parser.add_argument("--feature_dims", type=int, default=4096, help="latent feature dimension")
parser.add_argument("--num_workers", type=int, default=8, help='num of CPUs')
parser.add_argument("--backbone", type=str, default="vgg16", help='weight for heading loss')
parser.add_argument("--num_grd_image", type=int, default=7, help='number of ground view image (1, 3, 5, 7)')
parser.add_argument('--no_mask', default=False, action='store_true')
parser.add_argument('--MHA_layers', type=int, default=6, help="number of MHA layers")
parser.add_argument('--nHeads', type=int, default=8, help="number of heads")

opt = parser.parse_args()
print(opt)

#set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

length = 7

transMixer = TransMixer(transDimension=opt.feature_dims, max_length=length, numLayers = opt.MHA_layers, nHead=opt.nHeads)
grdFeatureExtractor = StreetFeatureExtractor(backbone = opt.backbone)
satelliteFeatureExtractor = SatelliteFeatureExtractor(backbone = opt.backbone, inputChannel=3)


#load model
epoch = f"epoch_{opt.epoch}"
transMixerCkpt = torch.load(os.path.join(opt.model_folder, "training_logs", epoch, f'trans_{opt.epoch}.pth'))
grdFeatureExtractorCkpt = torch.load(os.path.join(opt.model_folder, "training_logs", epoch, f'GFE_{opt.epoch}.pth'))
satelliteFeatureExtractorCkpt = torch.load(os.path.join(opt.model_folder, "training_logs", epoch, f'SFE_{opt.epoch}.pth'))


transMixer.load_state_dict(transMixerCkpt)
grdFeatureExtractor.load_state_dict(grdFeatureExtractorCkpt)
satelliteFeatureExtractor.load_state_dict(satelliteFeatureExtractorCkpt)

#all networks to cuda if available

transMixer.to(device)
grdFeatureExtractor.to(device)
satelliteFeatureExtractor.to(device)


#data loader

transformsStreet = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
                ]
transformsSatellite = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
                ]

dataset = ImageDataset(transforms_street=transformsStreet,transforms_sat=transformsSatellite,mode="val",zoom=20)
Loader = DataLoader(dataset,batch_size=opt.batch_size, shuffle=False, num_workers= 8)


#set the model to evaluate mode
transMixer.eval()
grdFeatureExtractor.eval()
satelliteFeatureExtractor.eval()

valSateFeatures = None
valStreetFeature = None
valHeadingError = 0
valLocError = 0
with torch.no_grad():
    for i, batch in enumerate(tqdm(Loader)):

        sateFeature, grdGlobalLatent=\
        InferOnce(grdFeatureExtractor,\
        satelliteFeatureExtractor, \
        transMixer, batch, device, opt.no_mask)


        #softmargin triplet loss
        sateFeatureUnit = sateFeature / torch.linalg.norm(sateFeature,dim=1,keepdim=True)
        grdGlobalLatentUnit = grdGlobalLatent / torch.linalg.norm(grdGlobalLatent,dim=1,keepdim=True)

        #stack features to the container
        if valSateFeatures == None:
            valSateFeatures = sateFeatureUnit.detach()
        else:
            valSateFeatures = torch.cat((valSateFeatures, sateFeatureUnit.detach()), dim=0)

        if valStreetFeature == None:
            valStreetFeature = grdGlobalLatentUnit.detach()
        else:
            valStreetFeature = torch.cat((valStreetFeature, grdGlobalLatentUnit.detach()), dim=0)


#Retrival accuracy
valAcc = ValidateAll(valStreetFeature, valSateFeatures)

print('top1', ':', valAcc[0, 1]*100)
print('top5', ':', valAcc[0, 5]*100)
print('top10', ':', valAcc[0, 10]*100)
print('top1%', ':', valAcc[0, -1]*100)


