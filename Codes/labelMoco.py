#encoding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from network import MainModel

class MoCo(nn.Module):
    def __init__(self,input_channel,num_class,base_econder,dim=128,K=8192,m=0.999,T=0.07) -> None:
        super(MoCo,self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_econder(input_channel=input_channel,num_class=num_class,num_blocks=1)
        self.encoder_k = base_econder(input_channel=input_channel,num_class=num_class,num_blocks=1)
        
        for param_q, param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.require_grad = False 

        self.register_buffer("dataqueue",torch.rand(dim,K))
        self.register_buffer("labelqueue",torch.zeros(K,dtype=torch.long))
        self.register_buffer("queue_ptr",torch.zeros(1,dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self): 
        '''
        momentum update of the key encoder
        '''
        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)  
    @torch.no_grad()
    def _dequeue_and_enqueue(self,keys,keyslabel):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0 
        self.dataqueue[:,ptr:ptr+batch_size] = keys.T
        self.labelqueue[ptr:ptr+batch_size] = keyslabel 
        ptr = (ptr + batch_size) % self.K 
        self.queue_ptr[0] = ptr

    def forward(self,im_q,im_k,q_label,is_train=True):
        cls_q,q = self.encoder_q(im_q) 
        if is_train:
            with torch.no_grad(): 
                self._momentum_update_key_encoder() 
                _,k = self.encoder_k(im_k) 
            queRes = torch.einsum("nc,ck->nk",[q,self.dataqueue.clone().detach()])
            quelabel = self.labelqueue.clone().detach() #shape K

            quelabel = torch.stack([quelabel for i in range(queRes.shape[0])],dim=0)

            predictlabel = torch.where(quelabel == q_label.reshape(-1,1),1,-1)
            predictlabel = predictlabel.float()


            self._dequeue_and_enqueue(k,q_label)

            return cls_q,q,queRes,predictlabel
        else:

            return cls_q
    







