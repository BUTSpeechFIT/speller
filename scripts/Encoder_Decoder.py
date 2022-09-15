#! /usr/bin/bash

import sys
import torch
import torch.nn as nn
import os
from os.path import join, isdir

import kaldi_io
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import numpy as np




#sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from CMVN import CMVN
from utils__ import weights_init,count_parameters,weights_init_tanh
#print("this is Encoder_Decoder_2lr")
#sys.path.insert(0,'/mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/Basic_Attention_V1')
#print(args.Dec_model)

class Encoder_Decoder(nn.Module):
        def __init__(self,args):
                super(Encoder_Decoder, self).__init__()
                #print(args.Dec_model)
                #--------------------------------------
                # for adding the Transformwr class
                if 1:
 #                       sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
                        #--------------------------------------
                        if args.Dec_model=='Dec_2':
                                print('using Dec_2')
                                from Decoder_V2 import decoder 
                        #--------------------------------------
                        elif args.Dec_model=='Dec_3':
                                print('using Dec_3')
                                from Res_LSTM_Encoder_arg import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3 import decoder 
                        #--------------------------------------
                        elif args.Dec_model=='Dec_1_Pret':
                                print('using Dec_1_Pret')
                                from Res_LSTM_Encoder_arg import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1_Pret import decoder
                        #-------------------------------------- 
                        elif args.Dec_model=='Dec_1_L2':
                                print('using Dec_1_L2')
                                from Res_LSTM_Encoder_arg import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1_2L import decoder
                        #--------------------------------------
                        elif args.Dec_model=='Dec_V1_WLSTM':
                                print('using_Decoder_V1_WordLSTM')
                                from Res_LSTM_Encoder_arg import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1_WordLSTM import decoder 
                        #--------------------------------------
                        #--------------------------------------
                        elif args.Dec_model=='subsamp_maxpool_dec_drop':
                                print('subsamp_maxpool_dec_drop')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1_dropput import decoder
                        #--------------------------------------
                        #--------------------------------------
                        elif args.Dec_model=='subsamp_maxpool':
                                print('using_subsamp_maxpool')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1 import decoder
                        #--------------------------------------
                        #--------------------------------------
                        elif args.Dec_model=='subsamp_maxpool_doubledrop':
                                print('using_subsamp_maxpool_doubledrop')
                                from Res_LSTM_Encoder_arg_maxpool_doubledrop import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1 import decoder
                        #--------------------------------------
                        #--------------------------------------
                        elif args.Dec_model=='subsamp_maxpool_projdrop':
                                print('using_subsamp_maxpool_projdrop')
                                from Res_LSTM_Encoder_arg_maxpool_projdrop import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1 import decoder
                        #--------------------------------------
                        #--------------------------------------
                        elif args.Dec_model=='Dec_3_temp':
                                print('using Dec_3')
                                from Res_LSTM_Encoder_arg import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_temp import decoder 
                        #--------------------------------------
                        #--------------------------------------
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_iter':
                                print('subsamp_maxpool_dec_drop_speller_iter')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_sp_iter import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_iter_nonshared':
                                print('subsamp_maxpool_dec_drop_speller_iter_nonshared')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_sp_iter_nonshared import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_iter_emb_400':
                                print('subsamp_maxpool_dec_drop_speller_iter_emb_400')
                                from Res_LSTM_Encoder_arg_maxpool_emb_400 import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_sp_iter_emb_400 import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_iter_bn':
                                print('subsamp_maxpool_dec_drop_speller_iter_bn')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_sp_iter_bn import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_bn_embs_ci_si':
                                print('subsamp_maxpool_dec_drop_speller_bn_embs_ci_si')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_bn_embs_ci_si import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller':
                                print('subsamp_maxpool_dec_drop_speller')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_embs_ci_si':
                                print('subsamp_maxpool_dec_drop_speller_embs_ci_si')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_embs_ci_si import decoder
                        elif args.Dec_model=='sp_b_concat':
                                print('sp_b_concat')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_b_concat import decoder
                        elif args.Dec_model=='sp_b_concat_yout':
                                print('sp_b_concat_yout')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_b_concat_yout import decoder
                        elif args.Dec_model=='sp_c_concat_subst':
                                print('sp_c_concat_subst')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_c_concat_subst import decoder
                        elif args.Dec_model=='sp_c_concat_subst_2':
                                print('sp_c_concat_subst_2')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_c_concat_subst_2 import decoder
                        elif args.Dec_model=='sp_d_concat_subst_lse10':
                                print('sp_d_concat_subst_lse10')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_d_concat_subst_lse10 import decoder
                        elif args.Dec_model=='sp_e_yout':
                                print('sp_e_yout')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_e_yout import decoder
                        elif args.Dec_model=='speller_mult_oovs':
                                print('speller_mult_oovs')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder import decoder
                        elif args.Dec_model=='speller_prob_oov_emb':
                                print('speller_prob_oov_emb')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V4_debug import decoder
                        elif args.Dec_model=='speller_mult_oovs_two_spellers':
                                print('speller_mult_oovs_two_spellers')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V5 import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_embs_ci_si_alter':
                                print('subsamp_maxpool_dec_drop_speller_embs_ci_si_alter')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_embs_ci_si_alter import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_embs_ci_si_noemb':
                                print('subsamp_maxpool_dec_drop_speller_embs_ci_si_noemb')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_ci_si_noemb import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_embs_ci':
                                print('subsamp_maxpool_dec_drop_speller_embs_ci')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_embs_ci import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_embs_si':
                                print('subsamp_maxpool_dec_drop_speller_embs_si')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_embs_si import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_si':
                                print('subsamp_maxpool_dec_drop_speller_si')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_si import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_ci':
                                print('subsamp_maxpool_dec_drop_speller_ci')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_dropout_ci import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_multiple_oovembs':
                                print('subsamp_maxpool_dec_multiple_oovembs')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_multiple_OOVembs import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_mult_oovembs_max':
                                print('subsamp_maxpool_dec_mult_oovembs_max')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_multiple_OOVembs_max import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_mult_oovembs_more':
                                print('subsamp_maxpool_dec_mult_oovembs_more')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_multiple_OOVembs_more import decoder
                        elif args.Dec_model=='subsamp_maxpool_dec_prob_oovembs':
                                print('subsamp_maxpool_dec_prob_oovembs')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_multiple_probOOVembs import decoder
                        elif args.Dec_model=='sep_OOVembs':
                                print('sep_OOVembs')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_sep_OOVembs import decoder
                        elif args.Dec_model=='sep_OOVembs_concat':
                                print('sep_OOVembs_concat')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_sep_OOVembs_concat import decoder
                        elif args.Dec_model=='sep_OOVembs_test':
                                print('sep_OOVembs_test')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_sep_OOVembs_test import decoder
                        elif args.Dec_model=='sep_OOVembs_noip':
                                print('sep_OOVembs_noip')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_sep_OOVembs_noip import decoder
                        #--------------------------------------       
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_2attn':
                                print('subsamp_maxpool_dec_drop_speller_2attn')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_word_char import decoder
                        #--------------------------------------
                        #--------------------------------------       
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_speller_2attn_pre':
                                print('subsamp_maxpool_dec_drop_speller_2attn_pre')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_word_char_pre import decoder
                        #--------------------------------------
                        #Decoder_V1_dropput_LMLoss
                        #--------------------------------------       
                        elif args.Dec_model=='subsamp_maxpool_dec_drop_LMLoss':
                                print('subsamp_maxpool_dec_drop_LMloss')
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1_dropput_LMLoss import decoder
                        #--------------------------------------
                        elif args.Dec_model=='Decoder_V1_dropput_incdec':
                                print("Decoder_V1_dropput_incdec")
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1_dropput_incdec import decoder
                        
                        #--------------------------------------
                        elif args.Dec_model=='Decoder_V3_word_char_bound':
                                print("Decoder_V3_word_char_bound")
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_word_char_bound import decoder

                        #--------------------------------------
                        #Decoder_V1_dropput_lookforward.py
                        elif args.Dec_model=='Decoder_V1_dropput_lookforward':
                                print("Decoder_V3_word_char_bound")
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1_dropput_lookforward import decoder
                        #--------------------------------------

                        elif args.Dec_model=='Decoder_V3_word_char_bound_char':
                                print("#Decoder_V3_word_char_bound_char")
                                from Res_LSTM_Encoder_arg_maxpool import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V3_word_char_bound_char import decoder

                        else:
                                print('using_decoder_V1')
                                from Res_LSTM_Encoder_arg import Conv_Res_LSTM_Encoder as encoder
                                from Decoder_V1 import decoder
                        

                        #--------------------------------------
                        #--------------------------------------
                        self.model_encoder=encoder(args=args)
                        self.model_decoder=decoder(args=args)

                        ##initialize xavier uniform ####default initialize does not work "imay be liked to use some bormalization layers needed"
                        if (str(args.init_full_model)=='xavier_lin_relu'):
                            self.model_encoder.apply(weights_init)
                            self.model_decoder.apply(weights_init)
                            print("All the weights are deafult initilaized")
                        else:
                            pass;

                        print("encoder:=====>",(count_parameters(self.model_encoder))/1000000.0)
                        print("decoder:=====>",(count_parameters(self.model_decoder))/1000000.0)
                #==================================
                else:
                        print("------------------------------------------------->")
        #--------------------------------------
        def forward(self,input,teacher_force_rate,Char_target,Word_target,smp_trans_text,only_speller):
#forward(self,H,teacher_force_rate,Char_target,Word_target,L_text,self.only_speller):
                ###encoder of the model
                H = self.model_encoder(input) 
#                print("encoder done")
                ###Decoder of the model        
                Decoder_out_dict = self.model_decoder(H, teacher_force_rate, Char_target, Word_target, smp_trans_text, only_speller)
#                print("decoder done")
                return Decoder_out_dict
        #--------------------------------------
        def predict(self,feat_path,args):
                """Input is the path to smp_feat and args file and the it ptodices output_dict"""
                print("went to the decoder loop")
                with torch.no_grad():
                        #### read feature matrices 
                        smp_feat=kaldi_io.read_mat(feat_path)

                        if args.apply_cmvn:
                            smp_feat=CMVN(smp_feat)
                        
                        input=torch.from_numpy(smp_feat)       
                        input = Variable(input.float(), requires_grad=False).double().float()
                        input=input.unsqueeze(0)

                        #breakpoint()
                        print("args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen",args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen)
                        H=self.model_encoder(input)
                        #Output_dict=self.model_decoder.decode_with_beam_LM(H,args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen)
                        ins_pen=args.ins_pen
                        lm_coupling=args.lm_coupling
                        Output_dict=self.model_decoder.decode_with_beam_LM(H,args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen,ins_pen,lm_coupling)
                        return Output_dict



class double_linear_warmup(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, sp_optimizer, init_lr, final_lr, sp_final_lr, warmup_steps=4000, sp_warmup_steps=400, min_lr=0, half_period=5000, optimizer_style='newbob', start_decay=0):
        """****-----------****"""

        self.optimizer = optimizer
        self.sp_optimizer = sp_optimizer
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.sp_final_lr = sp_final_lr
        self.warmup_steps = warmup_steps
        self.sp_warmup_steps = sp_warmup_steps
        self.rate = (self.final_lr - self.init_lr) / float(warmup_steps)
        self.sp_rate = (self.sp_final_lr - self.init_lr) / float(sp_warmup_steps)

        self.step_num = 0
        self.sp_step_num = 0
        self.reduction_factor = 1
        self.min_lr = min_lr
        self.half_period = half_period
        self.start_decay = start_decay
 
        self.decay_lr = 0
        self.optimizer_style = optimizer_style

        if self.optimizer_style=='newbob':
                    self.decay_function = self.newbob
        elif self.optimizer_style=='exp_decay':
                    self.decay_function = self.exp_decay

    def zero_grad(self):
        self.optimizer.zero_grad()

    def sp_zero_grad(self):
        self.sp_optimizer.zero_grad()

    def step(self):

        self._update_lr()
        #------------
        self.optimizer.step()

    def sp_step(self):

        self._update_sp_lr()
        #------------
        self.sp_optimizer.step()
    #----------------------------------------------------
    def _update_lr(self):
        self.step_num += 1
        if (self.step_num < self.warmup_steps+1):
            lr = (self.init_lr + self.step_num * self.rate )

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        else:
                if self.optimizer_style=='newbob':
                    pass;

                elif  self.optimizer_style=='exp_decay':
                    #after warmup wait till the decay started and once decay started get lr from decay
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.decay_function()
                else:
                    pass;
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
#            print(lr)


    def _update_sp_lr(self):
#        print("update sp lr")
        self.sp_step_num += 1
        if (self.sp_step_num < self.sp_warmup_steps+1):
            lr = (self.init_lr + self.sp_step_num * self.sp_rate )

            for param_group in self.sp_optimizer.param_groups:
                param_group['lr'] = lr

        else:
                if self.optimizer_style=='newbob':
                    pass;

                elif  self.optimizer_style=='exp_decay':
                    #after warmup wait till the decay started and once decay started get lr from decay
                    for param_group in self.sp_optimizer.param_groups:
                        param_group['lr'] = self.decay_function()
                else:
                    pass;

    #--------------------------------------------------
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_step_num(self, step_num):
        self.step_num = step_num

    def set_sp_step_num(self, step_num):
        self.sp_step_num = step_num

    def reduce_learning_rate(self, k=2):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(param_group['lr']/k)

    def reduce_sp_learning_rate(self, k=2):
        for param_group in self.sp_optimizer.param_groups:
            param_group['lr'] = float(param_group['lr']/k)

    def print_lr(self):
        present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        return present_lr[0]

    def print_sp_lr(self):
        present_lr=[param_group['lr'] for param_group in self.sp_optimizer.param_groups]
        return present_lr[0]

    def exp_decay(self):
        """ This half lr to every half_period updates """
        y0 = self.final_lr
        HP = 1
        k = -1*np.log(2)/self.half_period
        new_lr = np.maximum(y0*np.exp(k*self.step_num), self.min_lr)
        return new_lr

    def newbob(self):
        """ This lr is reduced by setting reduction factor """
        new_lr = self.final_lr/self.reduction_factor
        return new_lr

#---------------------------------------------------------------------------------------

class Trapizoidal_warmup(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer,init_lr,high_lr,final_lr, warmup_steps=4000,steady_steps=8000,cooldown_steps=4000):
        self.optimizer = optimizer
        
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.high_lr = high_lr

        self.warmup_steps = warmup_steps
        self.steady_steps = steady_steps
        self.cooldown_steps = cooldown_steps

        self.rate = (self.high_lr - self.init_lr) / float(warmup_steps)
        self.decay_rate = (self.high_lr - self.final_lr) / float(cooldown_steps)

        self.step_num = 0
        self.reduction_factor=1
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        #-------------
        self._update_lr()
        #------------
        self.optimizer.step()

    #----------------------------------------------------
    def _update_lr(self):
        self.step_num += 1

        if self.step_num < self.warmup_steps:
            ##linar increase
            lr = (self.init_lr + self.step_num * self.rate )
        
        elif self.warmup_steps <= self.step_num <= (self.warmup_steps + self.steady_steps):
                ##constant
                lr=self.high_lr

        elif (self.warmup_steps + self.steady_steps) <= self.step_num <= (self.warmup_steps + self.steady_steps + self.cooldown_steps):                
                ###linear decay
                lr = (self.high_lr - (self.step_num - (self.warmup_steps + self.steady_steps))* self.decay_rate)

        elif self.step_num > (self.warmup_steps + self.steady_steps + self.cooldown_steps):
                lr=self.init_lr
        else:
            lr=self.init_lr

        #----------------------------------------------
        if lr < self.final_lr:
                lr=self.final_lr
            #print("lr is reduced to ----->",lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    #--------------------------------------------------
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_step_num(self, step_num):
        self.step_num=step_num

    def reduce_learning_rate(self, k=2):
        self.reduction_factor = self.reduction_factor*k
        #print(self.reduction_factor)

    def print_lr(self):
        present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        return present_lr[0]

class Linear_warmup(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, init_lr, final_lr, warmup_steps=4000, min_lr=0, half_period=5000, optimizer_style='newbob', start_decay=0):
        """****-----------****"""

        self.optimizer = optimizer
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.rate = (self.final_lr - self.init_lr) / float(warmup_steps)

        self.step_num = 0
        self.reduction_factor = 1
        self.min_lr = min_lr
        self.half_period = half_period
        self.start_decay = start_decay

        self.decay_lr = 0
        self.optimizer_style = optimizer_style

        if self.optimizer_style=='newbob':
                    self.decay_function = self.newbob
        elif self.optimizer_style=='exp_decay':
                    self.decay_function = self.exp_decay

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):

        self._update_lr()
        #------------
        self.optimizer.step()
    #----------------------------------------------------
    def _update_lr(self):
        self.step_num += 1
        if (self.step_num < self.warmup_steps+1):
            lr = (self.init_lr + self.step_num * self.rate )

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        else:
                if self.optimizer_style=='newbob':
                    pass;

                elif  self.optimizer_style=='exp_decay':
                    #after warmup wait till the decay started and once decay started get lr from decay
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.decay_function()
                else:
                    pass;
        #for param_group in self.optimizer.param_groups:
        #    lr = param_group['lr']
        #    print(lr)
    #--------------------------------------------------
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_step_num(self, step_num):
        self.step_num = step_num

    def reduce_learning_rate(self, k=2):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(param_group['lr']/k)

    def print_lr(self):
        present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        return present_lr[0]

    def exp_decay(self):
        """ This half lr to every half_period updates """
        y0 = self.final_lr
        HP = 1
        k = -1*np.log(2)/self.half_period
        new_lr = np.maximum(y0*np.exp(k*self.step_num), self.min_lr)
        return new_lr

    def newbob(self):
        """ This lr is reduced by setting reduction factor """
        new_lr = self.final_lr/self.reduction_factor
        return new_lr

#---------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
#sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
#import Attention_arg
#from Attention_arg import parser
#args = parser.parse_args()
#print(args)
#model,optimizer=Initialize_Att_model(args)
#Linopt=Linear_warmup(optimizer,init_lr,high_lr,final_lr, warmup_steps=4000,steady_steps=8000,cooldown_steps=4000)
#print(model)





