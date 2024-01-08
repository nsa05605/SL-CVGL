import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import MaxPool1d
from torch import einsum
import math
import random
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 8):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransMixer(nn.Module):
    def __init__(self, transDimension, nHead=8, numLayers=6, max_length = 7):
        '''
        Transformer for mixing street view features
        transDimension: transformer embedded dimension
        nHead: number of heads
        numLayers: number of encoded layers
        Return => features of the same shape as input
        '''
        super(TransMixer, self).__init__()
        self.dim = transDimension
        encoderLayer = nn.TransformerEncoderLayer(d_model = transDimension,\
            nhead=nHead, batch_first=True, dropout=0.3, norm_first = True)

        self.Transformer = nn.TransformerEncoder(encoderLayer, \
            num_layers=numLayers, \
            norm=nn.LayerNorm(normalized_shape=transDimension, eps=1e-6))

        self.positionalEncoding = PositionalEncoding(d_model = transDimension, max_len=max_length)
    
    def forward(self, x, pos=True, mask=False, masked_range = [0,6], max_masked = 3):
        def get_mask(x, training, max_masked = 3, rand_range=[0,6], special_mask = False):
            device = x.get_device()
            mask = torch.zeros((x.shape[0], x.shape[1]))
            mask = mask.to(device)
            if not training:
                if not special_mask:
                    return mask
                if max_masked == 7:
                    return mask
                elif max_masked == 5:
                    for i in [0,1]:
                        mask[:,i] = 1.0
                elif max_masked == 3:
                    for i in [0,1,2,3]:
                        mask[:,i] = 1.0
                elif max_masked == 1:
                    for i in [0,1,2,3,4,5]:
                        mask[:,i] = 1.0
                else:
                    raise RuntimeError("no such mask")
                return mask
            numMasked = random.randint(0, max_masked)
            if numMasked == 0:
                return mask
            else:
                masked_pos = np.random.choice(np.arange(rand_range[0], rand_range[1]+1), numMasked)
                for p in masked_pos:
                    mask[:, p] = 1.0
            return mask

        if pos:
            x = self.positionalEncoding(x)

        if mask:
            mask = get_mask(x, self.training, max_masked=max_masked, rand_range=masked_range)
            x = self.Transformer(x, src_key_padding_mask = mask)
            result = x[:, mask[0] != 1.0]
            result = result.reshape(x.shape[0], -1, x.shape[2])
        else:
            result = self.Transformer(x)

        return result

class StreetFeatureExtractor(nn.Module):
    def __init__(self, backbone="res18"):
        '''
        CNN for extracting street image feature
        outputSize: output feature size
        Return => extracted features of size #outputSize
        '''
        super(StreetFeatureExtractor, self).__init__()
        if backbone == "res18":
            bb = models.resnet18(pretrained=True)
            modules=list(bb.children())[:-1]    # bb(backbone)에서 마지막 레이어를 제외하고 불러옴
            self.dimAfterBB = 512 #feature dims after backbone
            self.featureExtractor=nn.Sequential(*modules)
        elif backbone == "vgg16":
            bb = models.vgg16(pretrained=True)
            self.dimAfterBB = 4096 #feature dims after backbone
            classifier = bb.classifier
            classifier = list(classifier)[:4]
            bb.classifier = nn.Sequential(*classifier)
            self.featureExtractor = bb
        elif backbone == "res34":
            bb = models.resnet34(pretrained=True)
            modules=list(bb.children())[:-1]
            self.dimAfterBB = 512 #feature dims after backbone
            self.featureExtractor=nn.Sequential(*modules)
        elif backbone == "res50":
            bb = models.resnet50(pretrained=True)
            modules=list(bb.children())[:-1]
            self.dimAfterBB = 2048 #feature dims after backbone
            self.featureExtractor=nn.Sequential(*modules)
        else:
            RuntimeError(f"not implemented this backbone {backbone}")

    def forward(self,x):
        x = self.featureExtractor(x)

        x = x.reshape(-1, self.dimAfterBB)

        return x

class SatelliteFeatureExtractor(nn.Module):
    def __init__(self, inputChannel = 6, backbone="res18"):
        '''
        CNN for extracting satellite image feature
        inputChannel: number of channels input image
        outputSize: output feature size
        Return => extracted features of size #outputSize
        '''
        super(SatelliteFeatureExtractor, self).__init__()
        if backbone == "res18":
            bb = models.resnet18(pretrained=True)
            self.dimAfterBB = 512 #feature dims after backbone
        elif backbone == "vgg16":
            bb = models.vgg16(pretrained=True)
            self.dimAfterBB = 4096 #feature dims after backbone
            classifier = bb.classifier
            classifier = list(classifier)[:4]
            bb.classifier = nn.Sequential(*classifier)
            self.featureExtractor = bb
        elif backbone == "res34":
            bb = models.resnet34(pretrained=True)
            self.dimAfterBB = 512 #feature dims after backbone
        elif backbone == "res50":
            bb = models.resnet50(pretrained=True)
            self.dimAfterBB = 2048 #feature dims after backbone
        else:
            RuntimeError(f"not implemented this backbone {backbone}")

        if backbone != 'vgg16':
            if inputChannel != 3:
                modules=list(bb.children())[1:-1]
                modules.insert(0, nn.Conv2d(6,64,7,stride=2,padding=3,bias=False))
            else:
                modules=list(bb.children())[:-1]
            self.featureExtractor=nn.Sequential(*modules)


    def forward(self,x):
        x = self.featureExtractor(x)
        x = x.reshape(-1, self.dimAfterBB)

        return x



if __name__ == "__main__":
    model_street = StreetFeatureExtractor(backbone="vgg16")

    print(model_street)

    feat = torch.rand((8, 3, 320, 180))

    print(model_street(feat).shape)


