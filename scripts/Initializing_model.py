#!/usr/bin/python
import sys
import os
from os.path import join, isdir
import torch
from torch import optim

from utils__ import weights_init,count_parameters
from Encoder_Decoder import Encoder_Decoder, Linear_warmup, double_linear_warmup, Trapizoidal_warmup
from Freezing_pretrained_weights import Freezing_pretrained_weights
#====================================================================================
def Initialize_Att_model(args):
               
        model = Encoder_Decoder(args)
        trainable_parameters=model.parameters()
        if args.optimizer=='adam':
                ## It is better genaralizable hen we train larges possible lr otherwise raining with small lr may lead to very bad convergence
                #[names of speller mdlules]
                optimizer=optim.Adam(params=trainable_parameters, lr=args.learning_rate, betas=(0.9, 0.99),amsgrad=True)

        elif args.optimizer=='Linear_warmup_adam':
                optimizer=optim.Adam(params=trainable_parameters, lr=args.learning_rate, betas=(0.9, 0.99),amsgrad=True)
                #==============================================================
                optimizer=Linear_warmup(optimizer,init_lr = args.init_lr,
                                        final_lr = args.learning_rate,
                                        warmup_steps = args.warmup_steps,
                                        min_lr = args.min_lr,
                                        half_period = args.half_period,
                                        optimizer_style = args.optimizer_style,
                                        start_decay = args.start_decay)
                #==============================================================

        elif args.optimizer=='Trapizoidal_warmup_adam':
                optimizer = optim.Adam(params=trainable_parameters, lr=args.learning_rate, betas=(0.9, 0.99),amsgrad=True)
                optimizer = Trapizoidal_warmup(optimizer,init_lr=args.init_lr,high_lr=args.learning_rate,final_lr=args.final_lr, warmup_steps=args.warmup_steps,steady_steps=args.steady_steps,cooldown_steps=args.cooldown_steps)

        elif args.optimizer=='adadelta':
                ###This was better than adam but less than adam with goood hyper parameters
                optimizer = optim.Adadelta(params=trainable_parameters,lr=args.learning_rate,rho=0.9, eps=1e-06, weight_decay=0)
        
        elif args.optimizer=='Linear_warmup_sgd':
                ###Any form of SGD did not converge , during the training process it showed the convergence and quickly the model stopped paying attention to encoder and 
                ###any form of worm up did not try
                optimizer = torch.optim.SGD(params=trainable_parameters, lr=args.learning_rate, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
                optimizer = Linear_warmup(optimizer,init_lr=args.init_lr, final_lr=args.learning_rate, warmup_steps=args.warmup_steps)

        elif args.optimizer=='Trapizoidal_warmup_sgd':
                optimizer = torch.optim.SGD(params=trainable_parameters, lr=args.learning_rate, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
                optimizer = Trapizoidal_warmup(optimizer,init_lr=args.init_lr,high_lr=args.learning_rate,final_lr=args.final_lr, warmup_steps=args.warmup_steps,steady_steps=args.steady_steps,cooldown_steps=args.cooldown_steps)
        
        elif args.optimizer=='double_linear_warmup_adam':
                optimizer=optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99),amsgrad=True)

                sp_optimizer=optim.Adam(params=model.parameters(), lr=args.sp_learning_rate, betas=(0.9, 0.99),amsgrad=True)
                optimizer=double_linear_warmup(optimizer,
                                        sp_optimizer,
                                        init_lr = args.init_lr,
                                        final_lr = args.learning_rate,
                                        sp_final_lr = args.sp_learning_rate,
                                        warmup_steps = args.warmup_steps,
                                        sp_warmup_steps = args.sp_warmup_steps,
                                        min_lr = args.min_lr,
                                        half_period = args.half_period,
                                        optimizer_style = args.optimizer_style,
                                        start_decay = args.start_decay)
        else:
                print("optimizer key word in valid")


        #=======================================
        strict_flag = True if args.strict_load_weights_flag else False
        pre_trained_weight = args.pre_trained_weight
        mode_of_loading_weight = args.mode_of_loading_weight
       
        model= model.cuda() if args.gpu else model
        model, optimizer = Freezing_pretrained_weights(model=model,
                                                optimizer=optimizer, pre_trained_weight=pre_trained_weight,
                                                strict_flag=strict_flag, mode_of_loading_weight=mode_of_loading_weight)

        print("model:=====>",(count_parameters(model))/1000000.0)
        model= model.cuda() if args.gpu else model
        return model, optimizer
#====================================================================================
#====================================================================================
"""
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
import Attention_arg
from Attention_arg import parser
args = parser.parse_args()
print(args)

args.optimizer='Linear_warmup_adam'
args.optimizer_style='newbob'
args.warmup_steps=30000
args.learning_rate=0.001

model,optimizer=Initialize_Att_model(args)
for i in range(1,35000):
        print(i,optimizer.step())
optimizer.reduce_learning_rate(2)
for i in range(1,5000):
        print(i,optimizer.step())
#print(model)
"""

