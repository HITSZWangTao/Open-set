#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from contentEn import PointFeature
from PointPositionEn import getAugResult
import math



class LSTMMeanModel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LSTMMeanModel, self).__init__()
        self.BiLSTM = nn.LSTM(input_size,hidden_size,num_layers=1,batch_first=True,bidirectional=True)
    def forward(self,X):
        self.BiLSTM.flatten_parameters()
        outdata,(hidden,cells) = self.BiLSTM(X)
        hiddens = hidden.transpose(1,0)
        hiddens = torch.sum(hiddens,dim=1)
        return hiddens

class LSTransBlock(nn.Module):
    def __init__(self,d_model,n_heads) -> None:
        super(LSTransBlock,self).__init__()
        self.translayer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
                                                     batch_first=True)
       
        self.BiLSTM = nn.LSTM(d_model,d_model//2,num_layers=1,batch_first=True,bidirectional=True)

    def forward(self,X):
        
        self.BiLSTM.flatten_parameters()
        res = self.translayer(X)

        outdata,(hidden,cells) = self.BiLSTM(res)
        hidden = torch.sum(hidden,dim=0)
        
        return outdata,hidden



class MainModel(nn.Module):
    def __init__(self,input_channel,num_class,num_blocks) -> None:
        super(MainModel,self).__init__()
        self.ContentEncoder = PointFeature(input_channel=input_channel)
        self.LSTransBlock1 = LSTransBlock(d_model=256,n_heads=4)
        self.predictLayers = nn.Sequential(nn.Linear(128*num_blocks,64),nn.ReLU(inplace=True),
                                           nn.Linear(64,num_class))

    def forward(self,Points):
        PointsY = Points[:,:,:,1]
        mindata, _ = torch.min(PointsY,dim=2)
        mindata, _ = torch.min(mindata,dim=1)
        Points[:,:,:,1] = Points[:,:,:,1] - mindata.reshape(-1,1,1)
        BatchRes = []
        for i in range(Points.shape[0]):
            AugPoints = getAugResult(Points[i])
            Feature = self.ContentEncoder(AugPoints)
            BatchRes.append(Feature)
        BatchRes = torch.stack(BatchRes,dim=0)
        outres1,hiddenres1 = self.LSTransBlock1(BatchRes)
        clres = self.predictLayers(hiddenres1)
        return clres,hiddenres1
        
        


