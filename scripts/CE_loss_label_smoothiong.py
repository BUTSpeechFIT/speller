#!/usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

#IGNORE_ID=5000
from torch.autograd import Variable


#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#===============================================
def cal_loss(pred, gold,IGNORE_ID,normalize_length,smoothing):
    """Calculate cross entropy loss, apply label smoothing if needed.  """
    normalize_length=True

    #import pdb;pdb.set_trace()
    if smoothing > 0.0:
        #import pdb;pdb.set_trace()
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        #print("eps,n_class,IGNORE_ID,,gold,gold.ne(IGNORE_ID).long(),gold_for_scatter,one_hot",eps,n_class,IGNORE_ID,gold,gold.ne(IGNORE_ID).long(),gold_for_scatter,one_hot)       
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)

        loss = loss.masked_select(non_pad_mask).sum() / n_word
        #print("n_word,loss",n_word,loss)

    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=IGNORE_ID,
                               reduction='elementwise_mean')
    return loss
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

def cal_performance(pred, gold,IGNORE_ID):
    """Calculate cross entropy loss, apply label smoothing if needed. Args: pred: N *C, score before softmax  gold: N """
    #import pdb;pdb.set_trace()
    #pred = pred.max(1)[1]
    #import pdb;pdb.set_trace()

    non_pad_mask = gold.ne(IGNORE_ID)
    #print(non_pad_mask,non_pad_mask.size(),non_pad_mask.sum())
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    n_correct=n_correct/float(non_pad_mask.sum())
    #print(len(n_correct),n_correct)
    #n_correct=n_correct.sum().item()/len(n_correct)
    n_correct=1.0-n_correct
    return n_correct

#=========================================================================================
def  CrossEntropyLabelSmooth(pred, gold,IGNORE_ID,normalize_length,smoothing,Ignore_padding=False):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        # unk_mask=targets.ne(self.unk_id)
       
        inputs  = pred
        targets = gold
        num_classes = pred.size(1)

        log_probs = F.log_softmax(inputs,dim=1)
        batch_size=inputs.size()[0]

        ##How scatter works here 
        #torch.zeros(log_probs.size())
        #vecotr.scatter_(dim,ind,value)
        #torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        targets = Variable(torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1))
        targets=targets.cuda() if log_probs.is_cuda else targets 
        targets = (1 - smoothing) * targets + smoothing/num_classes       
        loss = (- targets * log_probs).sum(dim=1)

        #------------------------------------------
        if Ignore_padding:
            ##zero the elements which belong to the padding id in the loss
            ####was needed for speller
            non_pad_mask = gold.ne(IGNORE_ID)
            loss=loss.masked_select(non_pad_mask)
        #------------------------------------------
        loss=loss.sum()

        return loss
#==============================================================================================
def  CrossEntropyLabelSmooth_charloss(pred, gold,IGNORE_ID,normalize_length,smoothing,Ignore_padding=False):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        # unk_mask=targets.ne(self.unk_id)
       
        inputs  = pred
        targets = gold
        num_classes = pred.size(1)

        log_probs = F.log_softmax(inputs,dim=1)
        batch_size=inputs.size()[0]

        ##How scatter works here 
        #torch.zeros(log_probs.size())
        #vecotr.scatter_(dim,ind,value)
        #torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        targets = Variable(torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1))
        targets=targets.cuda() if log_probs.is_cuda else targets 
        targets = (1 - smoothing) * targets + smoothing/num_classes       
        loss = (- targets * log_probs).sum(dim=1)
       
        #------------------------------------------
        #------------------------------------------
        return loss
