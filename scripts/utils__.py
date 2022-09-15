#!/usr/bin/python
#------------------------------------------------------------------
import torch
import torch.nn as nn
from torch import autograd, nn, optim
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import gc
import numpy as np
#===========================================================================================
#===========================================================================================
#===========================================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#===========================================================================================
#utils
def clip_gradients(model,clip_value):
        for param in model.parameters():
                #param.grad torch.clamp(param.grad, min=-0.5, max=0.5)
                param.grad.data.clamp_(min=-clip_value,max=clip_value)

#clip_gradients_max
def clip_gradients_max(model,clip_value):
        for param in model.parameters():
                #param.grad torch.clamp(param.grad, min=-0.5, max=0.5)
                param.grad.data.clamp_(max=clip_value)
#================================================================================
#================================================================================
#
def del_gar(interm_var):
        for var in interm_var:
                del var
        gc.collect()
#================================================================================
def epoch_initialize():
    train_cost=0
    val_cost=0
    spk_cost=0
    return train_cost,val_cost,spk_cost

def dSP(X):
    return X.data.numpy()

def saving_model(mdl,epoch,cost):
    if not isdir(model_dir):
        os.makedirs(model_dir)
    savepath=os.path.join(model_dir,epoch,cost+".model")
    print(savepath)
    torch.save(mdl,savepath)

#===============================================================================================
def xavier_uniform(m, gain):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(gain))
                if m.bias is not None:  
                        m.bias.data.fill_(0)

#================================================================================================
def weights_init(m):
        if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:  
                        m.bias.data.fill_(0)

        # if isinstance(m, nn.Conv1d):
        #         nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         if m.bias is not None:
        #                 m.bias.data.fill_(0)

        #if isinstance(m, nn.Embedding):
        #        m.weight.data.normal_(0, 1)
        
        ##added for the check
        # if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         if m.bias is not None:
        #                 m.bias.data.fill_(0)
#===============================================================================================
'''
def init_like_chainer(self):
        """Initialize weight like chainer.
        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """

        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()

                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1.0 / math.sqrt(n)
                    data.normal_(0, stdv)

                elif data.dim() in (3, 4):
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1.0 / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.0)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for i in range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[i].bias_ih)

'''

#===============================================================================================
def weights_init_tanh(m):
        if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                        m.bias.data.fill_(0)


        if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                        m.bias.data.fill_(0)

        ##added for the check
        ##i use conv2d with relu        
        if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                        m.bias.data.fill_(0)
#===============================================================================================
#===============================================================================================
def reduce_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
                lr=param_group['lr']
                param_group['lr'] = lr/2
                print("learning rate hse been reduced from ",lr,"to ",lr/2)     
#===============================================================================================
###########################################################
#==========================================================
def subsampling(a):
        a=a.squeeze(0)
        ln=a.size()[0]
        #print ln
        b=a if (ln%2==0) else a[:-1]
        #print b.size()
        c=torch.cat((b[::2],b[1::2]),1)
        #print c.size(),c.unsqueeze(0).size()
        #exit(0)
        return c.unsqueeze(0)
############################################################
def subsampling_2(a):
        ln=a.size()[1]

        b=a if (ln%2==0) else a[:,:-1,:]        

        c=torch.cat((b[:,::2,:],b[:,1::2,:]),2)

        return c
#===========================================================


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y
#====================================================
#=======================================================
def del_gar(interm_var):
        for var in interm_var:
                del var
        gc.collect()
#=======================================================
def read_as_list(input_list):
    with open(input_list) as f:
        content_spk = f.readlines()
        content_spk = [x.strip() for x in content_spk]
    return(content_spk)
#=======================================================
def add_gaussian_noise(smp_feat):
        mu=0
        sigma=0.075
        noise = np.random.normal(mu, sigma, [smp_feat.shape[0],1,smp_feat.shape[1]])
        smp_feat=smp_feat+noise
        return smp_feat.astype(np.float32)
#=======================================================
#===============================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#==============================================
#==============================================
#==============================================
def gaussian_noise(ins, gpu,mean=0, stddev=0.075):
    noise = torch.zeros(ins.size()).normal_(mean, stddev).float()
    noise = noise.cuda() if gpu else noise

    return ins.data + noise
#==============================================
#==============================================
#==============================================

#===================================
def batch_torch_subsamp(smp_feat):
        feat_L=smp_feat.size[1]

        if feat_L%2==0:
                smp_feat=smp_feat
        else:   
                z_b=smp_feat[-1,:,:].unsqueeze(0)
                #Z_b=np.zeros((smp_feat.shape[0],1,smp_feat.shape[2]),dtype=float)
                smp_feat=torch.cat((smp_feat,Z_b),1)
                smp_feat=torch.cat((smp_feat[:,::2,:],smp_feat[:,1::2,:]),2)
        return smp_feat
#====================================
def read_as_dict(input_list):
        with open(input_list) as f:
                content_dict={}
                rev_content_dict={}
                content_spk = f.readlines()
                content_spk = [x.strip() for x in content_spk]
                for x in content_spk:
                        content_dict[x.split()[0]]=x.split()[1]
                        rev_content_dict[x.split()[1]]=x.split()[0]
        return(content_dict,rev_content_dict)
#===================================================

#===================================================
#===================================================
def split_list(the_list, chunk_size):
    result_list = []
    while the_list:
        result_list.append(the_list[:chunk_size])
        the_list = the_list[chunk_size:]
    return result_list

#===================================================
#===================================================

import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
def plotting(name,attention_map):
         plt.figure(1,figsize=(50,50))
         plt.imshow(attention_map,interpolation='nearest')
         #plt.colorbar()
         plt.savefig(name,Bbox='tight',orientation='landscape')
         plt.close()
