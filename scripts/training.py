#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir
import numpy as np
import fileinput
from numpy.random import permutation
##------------------------------------------------------------------
import torch
import pdb

#**********************************************************************
####### Loading the Parser and default arguments
#import pdb;pdb.set_trace()
sys.path.insert(0,'/mnt/matylda6/iegorova/speller/scripts')
#from Set_gpus import Set_gpu

import Attention_arg
from Attention_arg import parser
args = parser.parse_args()
#if args.gpu:
#      Set_gpu()

#----------------------------
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
#----------------------------------------
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
os.environ['PYTHONUNBUFFERED'] = '1'
import glob
from statistics import mean
import json
import kaldi_io

#******************************************************************
###save architecture for decoding
model_path_name=join(args.model_dir,'model_architecture_')
with open(model_path_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

##setting the gpus in the gpu cluster
#**********************************
if args.gpu:
    from safe_gpu import safe_gpu
    gpu_owner = safe_gpu.GPUOwner()

#----------------------------------------------------------------
#=================================================================
from Dataloader_for_AM import DataLoader
from Initializing_model import Initialize_Att_model
from Load_sp_model import Load_sp_models
from Training_loop import train_val_model
from Spec_Augument import Spec_Aug_freqCont as Spec_Aug
from CMVN import CMVN
from utils__ import weights_init,reduce_learning_rate,read_as_list,gaussian_noise,plotting
from user_defined_losses import preprocess,compute_cer
from Decoding_loop import get_cer_for_beam 
from scheduled_sampling import Teacherforce_schedule
#===================================================================
if not isdir(args.model_dir):
        os.makedirs(args.model_dir)

png_dir=args.model_dir+'_png'
if not isdir(png_dir):
        os.makedirs(png_dir)
############################################
#=======================================================
def main():  
        ##
        teacher_force_rate_gen=Teacherforce_schedule(args)
        ##Load setpiece models for Dataloaders
        Word_model=Load_sp_models(args.Word_model_path)
        Char_model=Load_sp_models(args.Char_model_path)

        ###initilize the model
        model,optimizer=Initialize_Att_model(args)
        #============================================================
        #------------------------------------------------------------  
        train_gen = DataLoader(files=glob.glob(args.data_dir + "train_scp_splits/aa_*")+glob.glob(args.data_dir + "train_scp_splits_temp1/bb_*")+glob.glob(args.data_dir + "train_scp_splits_temp2/cc_*"),
                                max_batch_label_len=args.max_batch_label_len,
                                max_batch_len=args.max_batch_len,
                                max_feat_len=args.max_feat_len,
                                max_label_len=args.max_label_len,
                                Word_model=Word_model,
                                Char_model=Char_model,                        
                                apply_cmvn=int(args.apply_cmvn))    

        dev_gen = DataLoader(files=glob.glob(args.data_dir + "dev_scp"),
                                max_batch_label_len=args.max_batch_label_len,
                                max_batch_len=args.max_batch_len,
                                max_feat_len=5000,
                                max_label_len=1000,
                                Word_model=Word_model,
                                Char_model=Char_model,
                                apply_cmvn=int(args.apply_cmvn))
        #Flags that may change while training 
        weight_noise_flag=False
        spec_aug_flag=False
        val_history_WER=np.zeros(args.nepochs)   
        val_history_CER=np.zeros(args.nepochs)
        #======================================
        for epoch in range(args.nepochs):
            ####corect the flags later
                            
            if args.weight_noise_flag==2:
                    weight_noise_flag=True
                    #spec_aug_flag=True

            if args.spec_aug_flag==2:
                    spec_aug_flag=True            

            ##start of the epoch
            tr_WER=[]; tr_CER=[]; L_train_cost=[]
            model.train();
            args.only_speller = 0
            for trs_no in range(args.validate_interval):
                B1 = train_gen.next()
                assert B1 is not None, "None should never come out of the DataLoader"

                Output_trainval_dict=train_val_model(args = args, 
                                                    model = model,
                                                    optimizer = optimizer,
                                                    data_dict = B1,
                                                    weight_noise_flag=weight_noise_flag,
                                                    spec_aug_flag=spec_aug_flag,
                                                    trainflag = True,
                                                    teacher_force_rate_gen=args.teacher_force)

                
                #get the losses form the dict
                L_train_cost.append(Output_trainval_dict.get('cost_cpu'))
                tr_CER.append(Output_trainval_dict.get('Char_cer'))
                tr_WER.append(Output_trainval_dict.get('Word_cer'))
               # print(Output_trainval_dict.get('output_seq'))
                #==========================================
                if (trs_no%args.tr_disp==0):
                    print("tr ep:==:>",epoch,"sampl no:==:>",trs_no,"train_cost==:>",mean(L_train_cost),"WER:",mean(tr_WER),'CER:',mean(tr_CER))    
     
                    #------------------------
                    if args.plot_fig_training:
                        plot_name=join(png_dir,'train_epoch'+str(epoch)+'_attention_single_file_'+str(trs_no)+'.png')
                        #print(plot_name)
                        plotting(plot_name,attention_map)
            ###validate the model
            #=======================================================
            model.eval()
 #           args.only_speller = 0
            #=======================================================
            Vl_WER=[]; Vl_CER=[];L_val_cost=[]
            val_examples=0
            for vl_smp in range(args.max_val_examples):
                B1 = dev_gen.next()
                smp_feat = B1.get('smp_feat')
                val_examples+=smp_feat.shape[0]
                assert B1 is not None, "None should never come out of the DataLoader"

                ##brak when the examples are more
                if (val_examples >= args.max_val_examples):
                    break;
                #--------------------------------------                
                Val_Output_trainval_dict=train_val_model(args=args,
                                                        model = model,
                                                        optimizer = optimizer,
                                                        data_dict = B1,
                                                        weight_noise_flag=False,
                                                        spec_aug_flag=False,
                                                        trainflag = False,
                                                        teacher_force_rate_gen=0)
            
                L_val_cost.append(Val_Output_trainval_dict.get('cost_cpu'))
                Vl_CER.append(Val_Output_trainval_dict.get('Char_cer'))
                Vl_WER.append(Val_Output_trainval_dict.get('Word_cer'))
                attention_map=Val_Output_trainval_dict.get('attention_record').data.cpu().numpy()

                #======================================================     
                #======================================================
                if (vl_smp%args.vl_disp==0) or (val_examples==args.max_val_examples-1):
                    print("val epoch:==:>",epoch,"val smp no:==:>",vl_smp,"val_cost:==:>",mean(L_val_cost),"WER:",mean(Vl_WER),'CER:',mean(Vl_CER))    
                    if args.plot_fig_validation:
                        plot_name=join(png_dir,'val_epoch'+str(epoch)+'_attention_single_file_'+str(vl_smp)+'.png')
                        ##print(plot_name)                                    
                        plotting(plot_name,attention_map)                            
 
            #----------------------------------------------------
#==================================================================
            val_history_WER[epoch]=(mean(Vl_WER)*100)
            val_history_CER[epoch]=(mean(Vl_CER)*100)
            print("val_history WER:",val_history_WER[:epoch+1])
            print("val_history CER:",val_history_CER[:epoch+1])
            #================================================================== 
            ####saving_weights 
            ct="model_epoch_"+str(epoch)+"_sample_"+str(trs_no)+"_"+str(mean(L_train_cost))+"___"+str(mean(L_val_cost))+"__"+str(mean(Vl_CER))
            print(ct)
            
            torch.save(model.state_dict(),join(args.model_dir,str(ct)))
            torch.save(optimizer.state_dict(),join(args.model_dir,str(ct)+'_opt'))
#            torch.save(sp_optimizer.state_dict(),join(args.model_dir,str(ct)+'_sp_opt'))
            
            #######################################################                    
            #######################################################
            ###open the file write and close it to avoid delays
            with open(args.weight_text_file,'a+') as weight_saving_file:
                print(join(args.model_dir,str(ct)), file=weight_saving_file)

            with open(args.Res_text_file,'a+') as Res_saving_file:
                print(float(mean(Vl_CER)), file=Res_saving_file)
            #=================================
            
            #early_stopping and checkpoint averaging: 
            if args.reduce_learning_rate_flag:
#                if args.CER_hist:
#                  print("CER history")
                A=val_history_WER
#                else:
                B=val_history_CER
#                    print("WER history")
                Non_zero_loss=A[A>0]
                sp_non_zero_loss=B[B>0]
                #Non_zero_loss[:-1]-=0.5
                min_cpts=np.argmin(Non_zero_loss)
                sp_min_cpts=np.argmin(sp_non_zero_loss)
                Non_zero_len=len(Non_zero_loss)
                sp_non_zero_len=len(sp_non_zero_loss)
                
                if ((Non_zero_len-min_cpts) > 1) and epoch>args.lr_redut_st_th:

                    print("label_smoothing is switched_off")
                    model.model_decoder.label_smoothing = 0
                    
                    if 'warmup' in args.optimizer:
                        optimizer.reduce_learning_rate(2)
                    else:
                        reduce_learning_rate(optimizer)

                    ###start regularization only when model starts to overfit
                    weight_noise_flag=True
                    spec_aug_flag=True
            #    if ((sp_non_zero_len-sp_min_cpts) > 1) and epoch>args.lr_redut_st_th:
            #        if 'warmup' in args.sp_optimizer:
            #            sp_optimizer.reduce_sp_learning_rate(2)
            #        else:
            #            reduce_sp_learning_rate(sp_optimizer)
                #------------------------------------
                if 'warmup' in args.optimizer:
                    lr=optimizer.print_lr()
             #       sp_lr=optimizer.print_sp_lr()
                    print("learning rate of the epoch:",epoch,"is",lr) #, "sp lr", sp_lr)   
                else:
                    for param_group in optimizer.param_groups:
                        lr=param_group['lr']                
                        print("learning rate of the epoch:",epoch,"is",lr)   
#                if 'warmup' in args.sp_optimizer:
#                    lr=sp_optimizer.print_lr()
#                    print("speller learning rate of the epoch:",epoch,"is",lr)
#                else:
#                    for param_group in sp_optimizer.param_groups:
#                        lr=param_group['lr']
#                        print("speller learning rate of the epoch:",epoch,"is",lr)

                if args.early_stopping and epoch>args.lr_redut_st_th:
                    #------------------------------------
                    if (lr<=1e-8) or ((Non_zero_len-min_cpts) >= args.early_stopping_patience):
                        print("lr reached to a minimum value")
                        exit(0)                        
            #----------------------------------
            #**************************************************************
#=============================================================================================
#=============================================================================================
if __name__ == '__main__':
    main()



