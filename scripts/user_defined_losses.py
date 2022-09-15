from __future__ import absolute_import
import sys

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import editdistance

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision', 'CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss', 'RingLoss','compute_cer']

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, IGNORE_ID,epsilon=0.2, use_gpu=True,ignore_flag=False,UNK_ID=0):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.IGNORE_ID=IGNORE_ID
        self.unk_id=UNK_ID
        self.ignore_flag=ignore_flag
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        unk_mask=targets.ne(self.unk_id)
        
        log_probs = self.logsoftmax(inputs)
        batch_size=inputs.size()[0]

        targets = Variable(torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1))

        targets=targets.cuda() if log_probs.is_cuda else targets 
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        loss = (- targets * log_probs).sum(dim=1)
        if self.ignore_flag:
                loss = loss.masked_select(unk_mask)
        
        loss=loss.sum()
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

#-------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------
# class KLD_labsmooth(nn.Module):
#         def __init__(self,IGNORE_ID,label_smoothing=0):
            
#             super(KLD_labsmooth, self).__init__()
#             self.smoothing=label_smoothing
#             self.ignore_index=IGNORE_ID
#         def forward(self,pred, gold):
#                 x=pred
#                 target=gold
#                 criterion = nn.KLDivLoss(reduce=False)
#                 confidence = 1.0 - self.smoothing
#                 normalize_length=True
#                 #print(x.size())
#                 size=x.size(1)

#                 batch_size = x.size(0)
#                 x = x.view(-1, size)
#                 target = target.view(-1)
#                 with torch.no_grad():
#                     true_dist = x.clone()
#                     true_dist.fill_(self.smoothing / (size - 1))
#                     ignore = target == self.ignore_index  # (B,)
#                     #ignore = target == -1
#                     total = len(target) - ignore.sum().item()
#                     target = target.masked_fill(ignore, 0)  # avoid -1 index
#                     true_dist.scatter_(1, target.unsqueeze(1), confidence)
#                 kl = criterion(torch.log_softmax(x, dim=1), true_dist)
#                 #print(kl,true_dist,total,target,self.ignore_index,batch_size,ignore.sum().item())
#                 #print(kl.size(),true_dist.size(),total,target,self.ignore_index,batch_size)
#                 denom = total if normalize_length else batch_size

#                 return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
#-------------------------------------------------------------------------------------------------------------------------------------
# def cal_performance(pred, gold,IGNORE_ID):
#     """Calculate cross entropy loss, apply label smoothing if needed.
#     Args:
#         pred: N x T x C, score before softmax
#         gold: N x T
#     """
#     #IGNORE_ID=-1
#     # pred = pred.view(-1, pred.size(2))
#     # gold = gold.contiguous().view(-1)

#     # loss = cal_loss(pred, gold, smoothing)
#     #pred = pred.max(1)[1]
    
#     non_pad_mask =gold!=IGNORE_ID
#     non_pad_mask=non_pad_mask*1

#     #print(gold,IGNORE_ID,non_pad_mask)

#     n_correct =pred==gold
#     n_correct=n_correct*1
#     #print(n_correct)

#     n_correct = n_correct*non_pad_mask
#     n_correct=n_correct.sum()/non_pad_mask.sum()


#     n_correct=1.0-n_correct
#     return n_correct

#-------------------------------------------------------------------------------------------------------------------------------------

def cal_performance(pred, gold,IGNORE_ID):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """
    #IGNORE_ID=-1
    # pred = pred.view(-1, pred.size(2))
    # gold = gold.contiguous().view(-1)

    #print(pred,gold)
    # pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)
    #print(non_pad_mask,non_pad_mask.size(),non_pad_mask.sum())
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    n_correct=n_correct/float(non_pad_mask.sum())
    #print(len(n_correct),n_correct)
    #exit(0)
    #n_correct=n_correct.sum().item()/len(n_correct)
    n_correct=1.0-n_correct
    #exit(0)
    return n_correct

#--------------------
def compute_cer(label, pred,IGNORE_ID):
        #import pdb;pdb.set_trace()
        dist=0;

        #padding_len=(np.equal(label,IGNORE_ID)*1).sum()
        dist = editdistance.eval(label, pred)
        return float(dist)/len(label)
#------------------


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad



def preprocess(padded_input,IGNORE_ID,sos_id,eos_id):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
        #print(ys)
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([eos_id])
        sos = ys[0].new([sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        #print(ys_out)
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, eos_id)
        
        ys_out_pad = pad_list(ys_out, IGNORE_ID) ####original
        #print(ys_out_pad)
        #exit(0)
        #ys_out_pad = pad_list(ys_out, self.eos_id)######modified
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad







#-------------------------------------------------------------------------------------------------------------------------------------
class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self, weight_ring=1.):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.weight_ring = weight_ring

    def forward(self, x):
        l = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return l * self.weight_ring

if __name__ == '__main__':
        pass
