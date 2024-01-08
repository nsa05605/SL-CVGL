import argparse
import os
import numpy as np
import math
import itertools
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SatelliteFeatureExtractor, StreetFeatureExtractor, TransMixer
from dataset import ImageDataset
from SMTL import softMarginTripletLoss

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


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        '''
        linear decay LR scheduler
        n_epochs: number of total training epochs
        offset: train start epochs
        decay_start_epoch: epoch start decay
        '''
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def save_model(savePath, transMixer, sateFeature, strFeature, epoch):
    modelFolder = os.path.join(savePath, f"epoch_{epoch}")
    os.makedirs(modelFolder)
    torch.save(transMixer.state_dict(), os.path.join(modelFolder, f'trans_{epoch}.pth'))
    torch.save(sateFeature.state_dict(), os.path.join(modelFolder, f'SFE_{epoch}.pth'))
    torch.save(strFeature.state_dict(), os.path.join(modelFolder, f'GFE_{epoch}.pth'))
    # torch.save(HPEstimator.state_dict(), os.path.join(modelFolder, f'HPE_{epoch}.pth'))


def InferOnce(grdFE, satFE, transMixer, batch, device, noMask):
    grdImgs = batch["street"].to(device)
    sateImgs = batch["satellite"].to(device)

    numSeqInBatch = grdImgs.shape[0]

    #street view featuer extraction
    # 여기 grdImgs가 차원이 어떻게 나오는지 확인해보자
    grdImgs = grdImgs.view(grdImgs.shape[0]*grdImgs.shape[1],\
        grdImgs.shape[2],grdImgs.shape[3], grdImgs.shape[4])

    grdFeature = grdFE(grdImgs)
    grdFeature = grdFeature.view(numSeqInBatch, SEQUENCE_SIZE, -1)

    #satellite view feature extraction
    # sateImgs도 차원이 각각 뭐를 의미하는지 확인해보자
    sateImgs = sateImgs.view(sateImgs.shape[0], sateImgs.shape[1]*sateImgs.shape[2],\
        sateImgs.shape[3], sateImgs.shape[4])
    sateFeature = satFE(sateImgs)
    sateFeature = sateFeature.view(numSeqInBatch, -1)
    # print(sateFeature.shape)
   
    # mask는 sequential dropout 하는 마스크 말하는 듯
    if not noMask:
        grdMixedFeature = transMixer(grdFeature, mask=True, masked_range = [0,6], max_masked=opt.max_masked)
    else:
        grdMixedFeature = transMixer(grdFeature, mask=False, masked_range = [0,6])
    grdGlobalFeature = grdMixedFeature.permute(0,2,1)
    grdGlobalLatent = F.avg_pool1d(grdGlobalFeature, grdGlobalFeature.shape[2]).squeeze(2)

    return sateFeature, grdGlobalLatent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--epoch", type=int, default=0, help="number of epochs start training")
    parser.add_argument("--decay_epoch", type=int, default=30, help="number of epochs start decaying LR")
    parser.add_argument("--batch_size", type=int, default=24, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--save_name", type=str, default='SAVE_NAME', help='name of the model')
    parser.add_argument("--feature_dims", type=int, default=4096, help="latent feature dimension")
    parser.add_argument("--backbone", type=str, default="vgg16", help='weight for heading loss')
    parser.add_argument("--beta1", type=float, default=0.9, help='beta1 for adam')
    parser.add_argument("--beta2", type=float, default=0.999, help='beta2 for adam')
    parser.add_argument("--num_workers", type=int, default=12, help='num of CPUs')
    parser.add_argument("--lambda_SMTL", type=float, default=1.0, help='weight for triplet loss')
    parser.add_argument("--gamma", type=float, default=10.0, help='value for SMTL gamma')
    parser.add_argument("--weight_decay", type=float, default=1e-2, help='value for SMTL gamma')
    parser.add_argument('--no_mask', default=False, action='store_true')
    parser.add_argument('--MHA_layers', type=int, default=6, help="number of MHA layers")
    parser.add_argument('--max_masked', type=int, default=6, help="max masked frames")
    parser.add_argument('--nHeads', type=int, default=8, help="number of heads")

    opt = parser.parse_args()
    print(opt)
    zoom = 20   # satellite 기준 zoom level
                # 수가 커질수록 가까움

    print(f"zoom level:{zoom}")


    #saving path for training logs
    writer = SummaryWriter(opt.save_name)
    savePath=os.path.join(opt.save_name, 'training_logs')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    else:
        print("Note! Saving path existed !")

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)

    length = 7
    print("sequence length : ", length)
    transMixer = TransMixer(transDimension=opt.feature_dims, max_length=length, numLayers = opt.MHA_layers, nHead=opt.nHeads)
    # feature_dims는 출력되는 특징 차원을 의미하는 것 같고
    # MHA_layers는 모르겠음

    grdFeatureExtractor = StreetFeatureExtractor(backbone = opt.backbone)
    satelliteFeatureExtractor = SatelliteFeatureExtractor(backbone = opt.backbone, inputChannel=3)
    
    # 여긴 multi-GPU인 경우
    if torch.cuda.device_count() > 1:
        transMixer = nn.DataParallel(transMixer)
        grdFeatureExtractor = nn.DataParallel(grdFeatureExtractor)
        satelliteFeatureExtractor = nn.DataParallel(satelliteFeatureExtractor)

    #all networks to cuda if available
    transMixer.to(device)
    grdFeatureExtractor.to(device)
    satelliteFeatureExtractor.to(device)

    # Optimizers
    optimizer = torch.optim.Adam(itertools.chain(transMixer.parameters(),grdFeatureExtractor.parameters(), satelliteFeatureExtractor.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=1e-6)
    lrSchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    #data loader
    # ColorJitter는 밝기, 채도 등등을 바꿔주고
    # RandomErasing은 부분부분을 가려주는 듯
    transformsSatellite = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH)),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.1, scale=(0.1,0.2),value="random"),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]
    transformsStreet = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.1, scale=(0.1,0.2),value="random"),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
                    ]

    transformsStreetVal = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
                    ]
    transformsSatelliteVal = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
                    ]

    trainLoader = DataLoader(ImageDataset(transforms_street=transformsStreet,transforms_sat=transformsSatellite,mode="train",zoom=zoom),\
        batch_size=opt.batch_size, shuffle=True, num_workers= opt.num_workers)

    valLoader = DataLoader(ImageDataset(transforms_street=transformsStreetVal,transforms_sat=transformsSatelliteVal,mode="val",zoom=zoom),\
        batch_size=opt.batch_size, shuffle=False, num_workers= opt.num_workers)

    ##training
    allLosses = []
    print("start training...")
    for epoch in range(opt.n_epochs):
        #set the model to train mode
        transMixer.train()
        grdFeatureExtractor.train()
        satelliteFeatureExtractor.train()

        epochLoss = 0
        epochTripletLoss = 0

        for batch in tqdm(trainLoader, disable = False):
            if batch["street"].shape[0] < 2:
                continue

            ###############################################
            # loss 부분을 수정한다면, 여기부터 수정하면 될 것 같음
            # 일단 SeqGeM이나 SeqVLAD 가져올 때 여기로 가져오자

            sateFeature, grdGlobalLatent =\
            InferOnce(grdFeatureExtractor,\
            satelliteFeatureExtractor, \
            transMixer, batch, device, opt.no_mask)



            #softmargin triplet loss
            sateFeatureUnit = sateFeature / torch.linalg.norm(sateFeature,dim=1,keepdim=True)
            grdGlobalLatentUnit = grdGlobalLatent / torch.linalg.norm(grdGlobalLatent,dim=1,keepdim=True)

            lossTriplet = softMarginTripletLoss(sateFeatureUnit, grdGlobalLatentUnit, opt.gamma)


            loss = opt.lambda_SMTL * lossTriplet
            #optimize
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(itertools.chain(transMixer.parameters(),grdFeatureExtractor.parameters(), satelliteFeatureExtractor.parameters()),  1.0)
            optimizer.step()

            epochLoss += loss.item()
            epochTripletLoss += lossTriplet.item()
        #step learning rate
        lrSchedule.step()

        #calculate epoch average loss
        epochLoss = float(epochLoss) / float(len(trainLoader))
        epochTripletLoss = float(epochTripletLoss) / float(len(trainLoader))
        
        #add to all losses list
        allLosses.append(epochLoss)
        if epoch % 10 == 9:
            save_model(savePath, transMixer, satelliteFeatureExtractor,\
                grdFeatureExtractor, epoch+1)

        #set the model to evaluate mode
        transMixer.eval()
        grdFeatureExtractor.eval()
        satelliteFeatureExtractor.eval()

        valSateFeatures = None
        valStreetFeature = None

        with torch.no_grad():
            for batch in tqdm(valLoader, disable = False):

                sateFeature, grdGlobalLatent=\
                InferOnce(grdFeatureExtractor,\
                satelliteFeatureExtractor, \
                transMixer, batch, device, opt.no_mask)


                #softmargin triplet loss
                sateFeatureUnit = sateFeature / torch.linalg.norm(sateFeature,dim=1,keepdim=True)
                grdGlobalRespresentUnit = grdGlobalLatent / torch.linalg.norm(grdGlobalLatent,dim=1,keepdim=True)

                #stack features to the container
                if valSateFeatures == None:
                    valSateFeatures = sateFeatureUnit.detach()
                else:
                    valSateFeatures = torch.cat((valSateFeatures, sateFeatureUnit.detach()), dim=0)

                if valStreetFeature == None:
                    valStreetFeature = grdGlobalRespresentUnit.detach()
                else:
                    valStreetFeature = torch.cat((valStreetFeature, grdGlobalRespresentUnit.detach()), dim=0)


            #Retrival accuracy
            valAcc = ValidateAll(valStreetFeature, valSateFeatures)
            print(f"==============Summary of epoch {epoch} on validation set=================")
            try:
                #print epoch loss
                print("---------loss---------")
                print(f"Epoch {epoch} Loss {epochLoss}")
                print(f"triplet loss:{epochTripletLoss}")
                writer.add_scalars('losses',{
                    'epoch_loss':epochLoss,
                    'triplet_loss':epochTripletLoss
                }, epoch)
                print("----------------------")
                print('top1', ':', valAcc[0, 1] * 100.0)
                print('top5', ':', valAcc[0, 5] * 100.0)
                print('top10', ':', valAcc[0, 10] * 100.0)
                print('top1%', ':', valAcc[0, -1] * 100.0)
                writer.add_scalars('validation recall@k',{
                    'top 1':valAcc[0, 1],
                    'top 5':valAcc[0, 5],
                    'top 10':valAcc[0, 10],
                    'top 1%':valAcc[0, -1]
                }, epoch)
            except:
                print(valAcc)

            print("========================================================================")

    writer.close()#close tensorboard


