#! /usr/bin/python

import sys
import os
import os.path

from os.path import join, isdir
import torch
from torch import optim

def Freezing_pretrained_weights(model,optimizer,pre_trained_weight,strict_flag,mode_of_loading_weight=''):
        """Freeze the trained weights based on different options"""
        weight_flag=pre_trained_weight.split('/')[-1]
        print("Initial Weights",weight_flag)
         
        #==============================================================
        if weight_flag != '0':
                weight_file=pre_trained_weight.split('/')[-1]
                weight_path="/".join(pre_trained_weight.split('/')[:-1])
                enc_weight=join(weight_path,weight_file)
                model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=strict_flag)

                
                optimizer_name=join(weight_path,weight_file,'_opt')
                if os.path.isfile(optimizer_name):
                        optimizer.load_state_dict(torch.load(optimizer_name, map_location=lambda storage, loc: storage))
                
                ####weights are already loaded
                if mode_of_loading_weight=='freeze_trained_weights':
                        A=torch.load(enc_weight, map_location=lambda storage, loc: storage)
                        trained_weights=A.keys()
                        for param in model.named_parameters():
                                if param[0] in trained_weights:
                                        param[1].requires_grad=False
                                        print(param[0],param[1].requires_grad)

                #======
                if mode_of_loading_weight=='freeze_encoder':
                        for param in model.model_encoder.named_parameters():
                                        param[1].requires_grad=False
                                        print(param[0],param[1].requires_grad)

                #======
                if mode_of_loading_weight=='freeze_decoder':
                        for param in model.model_decoder.named_parameters():
                                        param[1].requires_grad=False
                                        print(param[0],param[1].requires_grad)

        return model,optimizer
