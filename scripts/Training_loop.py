#!/usr/bin/python

import sys
import os
import torch
import numpy as np
import pdb
#----------------------------------------

#=========================================================
def forword_and_update(trainflag, model, optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, clip_grad_norm, only_speller):
#        breakpoint()
        #pdb.set_trace()
#        Decoder_out_dict = model(input,teacher_force_rate,Char_target,Word_target,smp_trans_text)
        Decoder_out_dict = model(input,teacher_force_rate,Char_target,Word_target,smp_trans_text,only_speller)
#        print(Decoder_out_dict)
        #--------------------------------

        cost=Decoder_out_dict.get('cost')
        if trainflag:
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad_norm)
        #--------------------------------------
                if only_speller:
                     optimizer.sp_step()
                else:
       #              print("normal step")
                     optimizer.step()
        #--------------------------------------
        cost_cpu = cost.item()
#        print("GOT HERE")
#        print(cost_cpu)
        return Decoder_out_dict,cost_cpu
#=========================================================



#--------------------------------------


#sys.path.insert(0,'/mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/Basic_Attention_V1')
from Spec_Augument import Spec_Aug_freqCont as Spec_Aug
from utils__ import weights_init,gaussian_noise
#---------------------------------------
def train_val_model(**kwargs):

        args = kwargs.get('args')
#        print(args)
        model = kwargs.get('model')
        optimizer= kwargs.get('optimizer')
#        sp_optimizer= kwargs.get('sp_optimizer')
        teacher_force_rate_gen = kwargs.get('teacher_force_rate_gen')

        trainflag = kwargs.get('trainflag')
        update_flag = kwargs.get('update_flag')


        weight_noise_flag = kwargs.get('weight_noise_flag')
        spec_aug_flag = kwargs.get('spec_aug_flag')

#        B1 = kwargs.get('data_dict')
#        smp_feat = B1.get('smp_feat')
#        smp_char_label = B1.get('smp_char_label')
#        smp_word_label = B1.get('smp_word_label')
        if args.only_speller:
          smp_trans_text = []
          for line in open(kwargs.get('data_dict'), "r"):
#            if line[0] != '{' and line[0] != '<':
            smp_trans_text.append(line.split()[0])#[1:])
#          print(smp_trans_text)
          smp_feat = np.empty([1, 1, args.input_size])
          smp_char_label = []
          smp_word_label = []
        else:
          B1 = kwargs.get('data_dict')
          smp_feat = B1.get('smp_feat')
#          print(smp_feat)
          smp_char_label = B1.get('smp_char_label')
#          print(smp_char_label)
          smp_word_label = B1.get('smp_word_label')
          smp_trans_text = B1.get('smp_trans_text')  


        #################finished expanding the keyword arguments#########
        ##===========================================
        if trainflag and args.spec_aug_flag and spec_aug_flag:
               smp_feat_mask = Spec_Aug(smp_feat,args.min_F_bands,args.max_F_bands,args.time_drop_max,args.time_window_max)
               smp_feat = smp_feat * smp_feat_mask

        # #==========================================
        if trainflag and (args.weight_noise_flag) and weight_noise_flag:
                 with torch.no_grad():
                         params = list(model.parameters()) #+ list(model_decoder.parameters())
                         param = [gaussian_noise(param, args.gpu) for param in params]
        #============================================
        ###################################################################
        optimizer.zero_grad() 
        if args.optimizer=='double_linear_warmup_adam':
          optimizer.sp_zero_grad()
 #       sp_optimizer.zero_grad()
        input=torch.from_numpy(smp_feat).float()

        Char_target=torch.LongTensor(smp_char_label)
        Word_target=torch.LongTensor(smp_word_label)
        #-----------------------------------------------------------------
        input=input.cuda() if args.gpu else input
        teacher_force_rate = teacher_force_rate_gen if trainflag else 0

        #--------------------------------
#        print("Training loop running decoder")
#        Decoder_out_dict = model(input,teacher_force_rate,Char_target,Word_target,smp_trans_text,args.only_speller)
        #--------------------------------
#        cost=Decoder_out_dict.get('cost')
#        if args.only_speller:
#          print("cost " + str(cost))
#        if trainflag:
#          if args.only_speller:

#                cost.backward()
#                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)
                #torch.nn.utils.clip_grad_norm_(model.model_decoder.parameters(),args.clip_grad_norm)

                ###training with accumilating gradients
                #cost=cost/args.accm_grad
                #cost.detach()

                ####gradient accumilation
                # if(smp_no%args.accm_grad)==0:
                #if update_flag:
                #     optimizer.step()
                #     optimizer.zero_grad()
                # cost_cpu=cost.item()
        #--------------------------------------
#                optimizer.step()
        #--------------------------------------
#        cost_cpu = cost.item() 
        ###output a dict

        OOM=False
        if args.only_speller:
            if trainflag:
                Decoder_out_dict,cost_cpu=forword_and_update(trainflag, model, optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, args.clip_grad_norm, args.only_speller)
            else:
                with torch.no_grad():
                    Decoder_out_dict, cost_cpu=forword_and_update(trainflag, model, optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, args.clip_grad_norm, args.only_speller)
        else:
            if trainflag:
                try:
#                if args.only_speller:
#                    Decoder_out_dict,cost_cpu=forword_and_update(trainflag, model, sp_optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, args.clip_grad_norm, args.only_speller)
#                else:
#                    print(trainflag)
#                    print(model)
#                    print(optimizer)
#                    print(input)
#                    print(teacher_force_rate)

                    Decoder_out_dict,cost_cpu=forword_and_update(trainflag, model, optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, args.clip_grad_norm, args.only_speller)
#                    print(Decoder_out_dict)

                except Exception as e:
                   if 'CUDA out of memory' in str(e):
                      OOM=True
                      torch.cuda.empty_cache()
                      print("The model in OOM condition batch size for the batch is:", input.shape)
                   else:
                        ####print if some other error occurs
                        print("There is some other error",str(e))

            ###When there is oom eror make the batch size 2
                if OOM: #and not args.only_speller:
                    batch_size = input.shape[0]
                    input=input[:2]
                    Word_target = Word_target[:2]
                    Char_target = Char_target[:2]
                    smp_trans_text = smp_trans_text[:2]

                    print("The model running under OOM condition batch size for the batch is:", input.shape[0])
                    Decoder_out_dict, cost_cpu=forword_and_update(trainflag, model, optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, args.clip_grad_norm, args.only_speller)
    #        else:
#                print("OOM condition ignored because only speller")
     #           Decoder_out_dict, cost_cpu=forword_and_update(trainflag, model, optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, args.clip_grad_norm, args.only_speller)
        #---------------
            else:
                with torch.no_grad():
                    Decoder_out_dict, cost_cpu=forword_and_update(trainflag, model, optimizer, input,teacher_force_rate,Char_target,Word_target,smp_trans_text, args.clip_grad_norm, args.only_speller)


        attention_record = Decoder_out_dict.get('attention_record') #[:,:,0].transpose(0,1)
        #==================================================
        Output_trainval_dict={'cost_cpu':cost_cpu,
                            'attention_record':attention_record,
                            'Char_cer':Decoder_out_dict.get('Char_cer'),
                            'Word_cer':Decoder_out_dict.get('Word_cer'),
                            'PPL':Decoder_out_dict.get('PPL'),
                            'Char_Attention_record':Decoder_out_dict.get('Char_Attention_record')}
        return Output_trainval_dict
#=========================================================
