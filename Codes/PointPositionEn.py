#encoding=utf-8

import torch
import numpy as np
from pytorch3d.ops import knn_points,knn_gather


def getKnnRes(pointsAnchor,pointsLeft,pointsRight,K=3):
    '''
    input:
        points1: BatchSize,PointsNumber,Dimension(4) Anchor
        points2: BatchSize,PointsNumber,Dimension(4) Left Source
        points3: BatchSize,PointsNumber,Dimension(4) Right Source

        K: K near points
    output:
        Concact Result: Batchsize,PointsNumber, Dimension + K*4
    '''
    #idx: BatchSize,PointNumber,K, 
    _,leftidx,_ = knn_points(pointsAnchor[:,:,:3],pointsLeft[:,:,:3],K=K,return_nn=True)
    _,rightidx,_ = knn_points(pointsAnchor[:,:,:3],pointsRight[:,:,:3],K=K,return_nn=True)

    nn_gather_feature_left = knn_gather(pointsLeft,leftidx)
    nn_gather_feature_right = knn_gather(pointsRight,leftidx)
    
    
    return nn_gather_feature_left,nn_gather_feature_right

def getAugResult(BatchData):
    '''
    BatchData: BatchSize,PointNumber,Dimension(4)
    '''
    BatchSize,PointNumber,Dimension = BatchData.shape
    paddings = torch.zeros(size=[1,PointNumber,Dimension]).cuda()
    LeftData = torch.cat([paddings,BatchData[:BatchSize-1]],dim=0)
    RightData = torch.cat([BatchData[1:],paddings],dim=0)

    nn_gather_left,nn_gather_right = getKnnRes(BatchData,LeftData,RightData)
    BatchDataExpand = BatchData.unsqueeze(2).repeat(1,1,3,1)
    BatchLeft = (BatchDataExpand - nn_gather_left).reshape(BatchSize,PointNumber,-1) 
    result = torch.cat([BatchData,BatchLeft],dim=-1)

    return result


















if __name__ == "__main__":
    '''
    data = torch.tensor([[[1,2,3],[2,3,4],[3,4,15]],[[1,8,9],[2,6,7],[3,2,1]]]).float()
    data2 = torch.tensor([[[1,7,1],[2,8,2],[4,5,6]],[[3,4,4],[8,7,1],[2,3,4]]]).float()
    dist,idx,nn = knn_points(data,data2,K=1,return_nn=True)
    print(idx.shape)
    res = knn_gather(data2,idx)
    print(res.shape)
    '''
    data = torch.rand(size=[128,45,180,4])
    dataY = data[:,:,:,1]
    mindata,minIndex = torch.min(dataY,dim=2)
    mindata,_ = torch.min(mindata,dim=1)
    data[:,:,:,1] = data[:,:,:,1] - mindata.reshape(-1,1,1,)
    print(data.shape)




