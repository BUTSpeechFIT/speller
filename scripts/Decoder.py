#!/usr/bin/python
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as wtnrm
from statistics import mean



import numpy as np
# import keras
from keras.preprocessing.sequence import pad_sequences
#
import sys
#
from Load_sp_model import Load_sp_models
from CE_loss_label_smoothiong import cal_performance
from CE_loss_label_smoothiong import CrossEntropyLabelSmooth as cal_loss
from CE_loss_label_smoothiong import CrossEntropyLabelSmooth_charloss as char_loss
#
from user_defined_losses import preprocess,compute_cer
import sys
import os
from os.path import join, isdir
import glob
from get_speller_word_prior import Get_Priors
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==========================================================
#==========================================================
class decoder(nn.Module):
        def __init__(self, args):
                super(decoder, self).__init__()

                self.only_speller = args.only_speller

                #-------------------------------------------
                self.attention_type=args.attention_type
                self.use_speller = args.use_speller
                self.num_OOV_embs = args.num_OOV_embs                

                self.spell_everything = args.spell_everything
                self.cond_word_on_speller = args.cond_word_on_speller
                self.char_label_smoothing = args.char_label_smoothing
                self.speller_cost_weight = args.speller_cost_weight

                self.norm_spell_loss = args.norm_spell_loss
                self.reinitialize_speller_always = args.reinitialize_speller_always
                self.use_si_ci_spelling = args.use_si_ci_spelling
                self.word_prior_nomalize_speller = args.word_prior_nomalize_speller
                self.Lexume_layers = args.decoder_layers
                self.speller_input = args.speller_input

                #
                ###new_flags
                #self.initialize_speller_always=1
                #---------------------------------------------
                #self.use_speller = 1

                #
                self.use_word = args.use_word
                self.spell_loss_perbatch = args.spell_loss_perbatch
                self.ctc_target_type = args.ctc_target_type
                self.label_smoothing = args.label_smoothing

                self.Word_model = Load_sp_models(args.Word_model_path)
                self.Char_model = Load_sp_models(args.Char_model_path)
                #

                self.Word_model_path = args.Word_model_path
                self.Char_model_path = args.Char_model_path

                ####word model
                self.targets_no = int(self.Word_model.__len__())
                self.pad_index  = self.targets_no
                self.sos_id     = self.targets_no + 1
                self.eos_id     = self.targets_no + 2
                self.mask_id    = self.targets_no + 3
                self.Wout_size  = self.targets_no + 4 + self.num_OOV_embs 
                self.word_unk   = self.Word_model.unk_id()
                self.Word_SIL_tok   = self.Word_model.EncodeAsIds('_____')[0]
                ####Char model
                #---------------------------------------
                self.Ch_tgts_no     = int(self.Char_model.__len__())
                self.Char_pad_id    = self.Ch_tgts_no
                self.Char_sos_id    = self.Ch_tgts_no + 1
                self.Char_eos_id    = self.Ch_tgts_no + 2
                self.Char_mask_id   = self.Ch_tgts_no + 3
                self.Char_out_size  = self.Ch_tgts_no + 4
                self.Ch_SIL_tok   = self.Char_model.EncodeAsIds('_____')[0]
                #---------------------------------------
                self.use_gpu = args.gpu
                self.hidden_size = args.hidden_size                
                self.emb_dim = args.hidden_size                


                self.dropout_layer = nn.Dropout(p=args.dec_dropout)
                #---------------------------------------
                #Word_model layers
                #---------------------------------------

                #ATTENT parameters
                kernel_size = 11 #kernal is always odd
                padding     = (kernel_size - 1) // 2

                self.conv   = nn.Conv1d(1, self.hidden_size, kernel_size, padding=padding)
                self.PSI    = nn.Linear(self.hidden_size,self.hidden_size)
                self.PHI    = nn.Linear(self.hidden_size,self.hidden_size)
                self.attn   = nn.Linear(self.hidden_size,1) 
                #---------------------------------------

                list_of_attentions_hdim=['LAS_LOC']
                self.Lexume_Lstm_input= self.hidden_size*2 #if args.tie_dec_weights else self.hidden_size*2 #self.hidden_size if (self.attention_type in list_of_attentions_hdim ) else self.hidden_size*2
                #-----------------------------------------
                ####### Lexume-Lstm
                self.Lexume_Lstm    = nn.LSTM(self.Lexume_Lstm_input,self.hidden_size, self.Lexume_layers ,batch_first=False,bidirectional=False)#1
                self.Wembedding     = nn.Embedding(self.Wout_size, self.hidden_size)
                self.W_Dist_1         = nn.Linear(self.hidden_size*2,self.hidden_size)
                self.W_Dist_2         = nn.Linear(self.hidden_size,self.Wout_size)
#                self.Lexume_Lstm_input= self.hidden_size if (self.attention_type in list_of_attentions_hdim ) else self.hidden_size*2
                #-----------------------------------------
                ####### Lexume-Lstm
#                self.Lexume_Lstm    = nn.LSTM(self.Lexume_Lstm_input,self.hidden_size, self.Lexume_layers ,batch_first=False,bidirectional=False)#1
#                self.Wembedding     = nn.Embedding(self.Wout_size, self.hidden_size)
#                self.W_Dist         = nn.Linear(self.hidden_size*2,self.Wout_size)
                #---------------------------------------  
                if self.use_speller:
                    if self.speller_input == 'emb':
                        self.speller_input_dim = self.hidden_size #if self.use_si_ci_spelling else self.hidden_size*2
                    elif self.speller_input == 'emb_ci_si':
                        self.speller_input_dim = self.hidden_size*3
                    else:
                        self.speller_input_dim = self.hidden_size*2
                #    self.speller_input_dim = self.hidden_size*3 #if self.use_si_ci_spelling else self.hidden_size*2

                    self.Spelling_Lstm  = nn.LSTM(self.speller_input_dim, self.hidden_size, 1 ,batch_first=False,bidirectional=False)#1
                #    self.Ch_embedding   = nn.Embedding(self.Char_out_size, self.hidden_size)
                    self.Ch_Dist        = nn.Linear(self.hidden_size,self.Char_out_size)          
                #---------------------------------------
                #---------------------------------------
                self.log_softmax = nn.Softmax(dim=1)      
                self.relu        = nn.ReLU()
                self.softmax     = nn.Softmax(dim=0)
                self.tanh        = nn.Tanh()
                #---------------------------------------
                self.ctc_weight = args.ctc_weight
                self.compute_ctc = args.compute_ctc
                if args.tie_dec_weights:
                    self.Wembedding.weight = self.W_Dist_2.weight

                #breakpoint()
                if self.training and self.word_prior_nomalize_speller:
                    self.Word_priors = Get_Priors(self.Wout_size, glob.glob(args.data_dir + "train_scp"))
                    self.Word_priors = torch.from_numpy(self.Word_priors).float()
                    self.Word_priors = self.Word_priors.cuda() if args.gpu else self.Word_priors
                    self.Word_priors=self.Word_priors.squeeze()
                    self.Word_priors[self.word_unk] = 0
                    self.Word_priors /= self.Word_priors.sum() 
                    self.Word_priors[self.Word_priors==0]=1
                else:
                    # -----> For dummy value :
                    self.Word_priors=0
                #self.Word_priors
                #breakpoint()


                # if self.compute_ctc:
                #     if self.ctc_target_type=='word':
                #         self.CTC_output_layer = nn.Linear(self.hidden_size,self.Char_out_size)
                #         self.CTC_Loss = torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=True)
                    
                #     elif self.ctc_target_type=='char':
                #         self.CTC_output_layer = nn.Linear(self.hidden_size,self.Wout_size)
                #         self.CTC_Loss = torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=True)
                #     else:
                #         print("ctc_target_type given wrong",self.ctc_target_type)
                #         exit(0)
        #---------------------------------------
        #==============================================
        #----------------------------------------------
        def select_step(self, H, yi, hn1, cn1, si, alpha_i_prev, ci):
                    if self.attention_type=='LAS':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.step_LAS(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    elif self.attention_type=='Collin_monotonc':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.step_Collin_Raffel_monotonic(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    elif self.attention_type=='Location_aware':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.step_Location_aware(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    #------------------------------
                    elif self.attention_type=='LAS_LOC':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.LAS_LOC(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    #------------------------------
                    elif self.attention_type=='LAS_LOC_ci':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.LAS_LOC_ci(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    #------------------------------
                    #------------------------------
                    else:
                        print("-atention type undefined choose LAS |Collin_monotonc| Location_aware--->",self.attention_type)
                        exit(0)
                    return yout, alpha_i, si_out, hn1_out, cn1_out, ci_out
            #----------------------------------
        #===============================================
        def Additive_Att_LOC(self,H,si,alpha_i_prev):                        
                psi=self.dropout_layer(self.PSI(si))
                phi=self.dropout_layer(self.PHI(H))

                pfi=self.conv(alpha_i_prev.transpose(0,2)).transpose(0,2).transpose(1,2)
                
                ei=self.attn(self.tanh(phi + psi.expand_as(phi) + pfi))        
                
                alpha_i=self.softmax(ei.squeeze(2)).unsqueeze(1)
                
                ci=torch.bmm(alpha_i.transpose(2,0),H.transpose(0,1))
                ci=ci.transpose(0,1)

                ci_pl_si=torch.cat([ci,si],2)
                return ci,alpha_i,ci_pl_si,phi

        #===============================================
        def Additive_Att(self,H,si,alpha_i_prev):                        
                psi=self.dropout_layer(self.PSI(si))
                phi=self.dropout_layer(self.PHI(H))             
               
                ei=self.attn(self.tanh(phi + psi.expand_as(phi)))                        
                alpha_i=self.softmax(ei.squeeze(2)).unsqueeze(1)

                ci=torch.bmm(alpha_i.transpose(2,0),H.transpose(0,1))
                ci=ci.transpose(0,1)

                ci_pl_si=torch.cat([ci,si],2)
                return ci,alpha_i,ci_pl_si,phi
        #----------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------
        def step_Collin_Raffel_monotonic(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                ####LAS model has no LOC_ATT
                #  ci_out=Additive_Att(si,H); si_out=RNN(si, [yi,ci_out]);  yout=W_Dist([si_out,ci_out])
                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att(H,si,alpha_i_prev)
                lstm_input=self.dropout_layer(lstm_input)

                lstm_input=torch.cat([yi,ci_out],2)
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))
                si_out=self.dropout_layer(si_out)

                ci_pl_si=torch.cat([ci_out,si_out],dim=2)
                ci_pl_si=self.dropout_layer(ci_pl_si)

                emb=self.W_Dist_1(ci_pl_si)
                yout=self.W_Dist_2(emb)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out

        def step_LAS(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                ####LAS model has no LOC_ATT
                #  si_out=RNN(si, [yi,ci]); ci_out=Additive_Att(si_out,H); yout=W_Dist([si_out,ci_out])
                lstm_input=torch.cat([yi,ci],2)
                lstm_input=self.dropout_layer(lstm_input)
                
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))
                si_out=self.dropout_layer(si_out)

                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att(H,si_out,alpha_i_prev)
                ci_pl_si=self.dropout_layer(ci_pl_si)
#                yout=self.W_Dist(ci_pl_si)
                emb=self.W_Dist_1(ci_pl_si)
                yout=self.W_Dist_2(emb)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out

        #----------------------------------------------------------------------------------------
        def step_Location_aware(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                ####Ci is not used here
                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att_LOC(H,si,alpha_i_prev)
                lstm_input=torch.cat([yi,ci_out],2)
                
                lstm_input=self.dropout_layer(lstm_input)
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))

                si_out=self.dropout_layer(si_out)
                #### op is from si and ci_out  
                lstm_output=torch.cat([si,ci_out],2) ####this is different from others its not [si_out,ci_out]
                lstm_output=self.dropout_layer(lstm_output)
                emb=self.W_Dist_1(ci_pl_si)
                yout=self.W_Dist_2(emb)
#                yout=self.W_Dist(lstm_output)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out
        #----------------------------------------------------------------------------------
        def LAS_LOC(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                #stm_input=torch.cat([yi,yi],dim=2)

                lstm_input=yi
                lstm_input=self.dropout_layer(lstm_input)

                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))
                si_out=self.dropout_layer(si_out)

                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att_LOC(H,si_out,alpha_i_prev)
                ci_pl_si=self.dropout_layer(ci_pl_si)
                emb=self.W_Dist_1(ci_pl_si)
                yout=self.W_Dist_2(emb)
 #               yout=self.W_Dist(ci_pl_si)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out
        #----------------------------------------------------------------------------------
        def LAS_LOC_ci(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                lstm_input=torch.cat([yi,ci],dim=2)
                lstm_input=self.dropout_layer(lstm_input)

                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))
                si_out=self.dropout_layer(si_out)

                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att_LOC(H,si_out,alpha_i_prev)
                ci_pl_si=self.dropout_layer(ci_pl_si)
                emb=self.W_Dist_1(ci_pl_si)
                yout=self.W_Dist_2(emb)
#                yout=self.W_Dist(ci_pl_si)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out
        #----------------------------------------------------------------------------------
        #==================================================================================
        #----------------------------------------------------------------------------------
        def Spelling_step(self,H,yi,hn1,cn1,si,ci, alpha_i_prev): #### This is just an alternative word LSTM but does not look in to encoder
              
                if self.reinitialize_speller_always:
                        batch_size = H.size(1)
                        AUCO_SEQ_len = H.size(0)
                        _, _, _, hn1, cn1, _, _ =self.initialize_decoder_states(batch_size,AUCO_SEQ_len)
                        # 
                        if hn1.size(0)>1:
                            hn1, cn1 = hn1[-1].unsqueeze(0), cn1[-1].unsqueeze(0)                 
                        #
                si_out,(hn1_out,cn1_out) = self.Spelling_Lstm(yi,(hn1,cn1))
                yout = self.Ch_Dist(si_out)
                alpha_i = alpha_i_prev ####to keep the location attention intact as it does not depend on encoder
                ci_out = ci
                ####Add attentipo
                #
                return yout,alpha_i,si_out,ci_out,hn1_out,cn1_out
        #=======================================================================================
        #-------------------------------------------
        def predict_from_speller(self, H, hn1, cn1, si, alpha_i_prev,ci,greedy_label):
                #breakpoint()

                maximum_char_lstm_len=20
                ###check this 
                if hn1.size(0)>1:
                    hn1, cn1 = hn1[-1], cn1[-1]
                #-------------------
                ###repeat the embeding vector equal to length of char string
                ##embedding that corrspods to the labels 
                ####i suspect this way of geting embeding probanbly i should be usding the preembedding  ci_pl_si, let us see

                yi              = self.Wembedding(greedy_label)
                if self.speller_input == 'emb':
                    Unk_emb_vec = yi
                elif self.speller_input == 'emb_ci_si':
                    Unk_emb_vec     = torch.cat([yi,ci,si],dim=2)
                elif self.speller_input == 'emb_ci':
                    Unk_emb_vec     = torch.cat([yi,ci],dim=2)
                elif self.speller_input == 'emb_si':
                    Unk_emb_vec     = torch.cat([yi,si],dim=2)
                else:
                    print("unknown speller input")
                    exit(0)
#                Unk_emb_vec     = torch.cat([yi,ci,si],dim=2) if self.use_si_ci_spelling else yi    ####output of word lstm

                Unk_emb_vec     = Unk_emb_vec.cuda() if self.use_gpu else Unk_emb_vec
                yi_c = torch.repeat_interleave(Unk_emb_vec, maximum_char_lstm_len, dim=0)

                ####Spelling Attention or no attention speller code 
                char_yout , alpha_i, si_out_char,ci_out_char, hn1_out, cn1_out = self.Spelling_step(H, yi_c, hn1, cn1, si, ci, alpha_i_prev)
                char_yout    = torch.transpose(char_yout, 0, 1)
                ######need a kind of beam search here
                #breakpoint()
                greedy_char_label=torch.argmax(F.softmax(char_yout,dim=2),dim=2)

                #
                # #============================================================================
                if self.spell_everything:
                    utt_with_unk = (greedy_label == greedy_label)
                    ##All Ture
                else:
                        ##Some Ture
                        utt_with_unk = (greedy_label != self.word_unk)
                        ##only makes sence to change when all the unk labels are spelled
                        if self.cond_word_on_speller:
                            utt_with_unk_vec   = utt_with_unk.unsqueeze(2).expand_as(hn1)*1
                            hn1_out = hn1 * (1-utt_with_unk_vec) + (utt_with_unk_vec) * hn1_out
                            cn1_out = cn1 * (1-utt_with_unk_vec) + (utt_with_unk_vec) * cn1_out
                #============================================================================
                ########
                char_lstm_loss=0
                Speller_cer=0
                Posterior_matrix = char_yout

                #self.get_charecters_for_sequences(OP['yseq'],Tok_model,Tok_padding_id,Tok_eos_id,Tok_model.unk_id())
                #breakpoint()
                # #-----------------------
                # #===============================================
                # ### Convert the hyps to text and add the text seq to dict
                # #breakpoint()
                #output_dict=[]
                #for OP in nbest_hyps:
            
                #OP['Text_seq'] = self.get_charecters_for_sequences(greedy_char_label, self.Char_model, self.Char_pad_id, self.Char_eos_id, self.Char_model.unk_id())
                #    output_dict.append(OP)
                # #-------------------------------------
                # #-------------------------------------
                output_dict=[]
                char_output_beam_hyp={}
                char_output_beam_hyp['Text_seq'] = self.get_charecters_for_sequences(greedy_char_label, self.Char_model, self.Char_pad_id, self.Char_eos_id, self.Char_model.unk_id())
                output_dict.append(char_output_beam_hyp)
                #breakpoint()
                # ###to slow
                # char_output_beam_hyp=self.Non_Autoreg_beamsrearch(Posterior_matrix=char_yout,
                #                         Tok_model=self.Char_model,
                #                         Tok_padding_id=self.Char_pad_id,
                #                         Tok_sos_id=self.Char_sos_id,
                #                         Tok_eos_id=self.Char_eos_id,
                #                         Tok_unk_id=self.Char_model.unk_id())
                
                #breakpoint()
                return output_dict
        #---------------------------------------------------------------------------------------
        def Spell_Rnn(self,UNK_TEXT_BPE, UNK_TEXT, H, hn1, cn1, si, alpha_i_prev,ci,greedy_label,yout):
                batch_size=H.size(1)
                ###check this###just use the last layer 
                if hn1.size(0)>1:
                        hn1, cn1 = hn1[-1].unsqueeze(0), cn1[-1].unsqueeze(0)


                #---------------------------------------------------------------------------------------
                #---------------------------------------------------------------------------------------
                if '_word.model' in self.Word_model_path:
                    ## get the charecters for the text
                    char_seq_list     = [self.Char_model.EncodeAsIds(i) for i in UNK_TEXT ]
                    char_seq_len_list = [len(self.Char_model.EncodeAsIds(i)) for i in UNK_TEXT ]  
                else:

                    char_seq_list     = [self.Char_model.EncodeAsIds(self.Word_model.DecodeIds([i])) if i < self.pad_index else [self.Char_pad_id] for i in UNK_TEXT_BPE ]
                    char_seq_len_list = [len(i) for i in char_seq_list ]  
                #---------------------------------------------------------------------------------------
                #---------------------------------------------------------------------------------------

                ###make batch of charecters
                UNK_CHAR_SEQ = pad_sequences(char_seq_list, maxlen=max(char_seq_len_list) ,dtype='int32',padding='post',value=self.Char_pad_id)

                #-------------------
                Target_UNK_CHAR_SEQ = Variable(torch.IntTensor(UNK_CHAR_SEQ), requires_grad=False).contiguous().long();
                Target_UNK_CHAR_SEQ = Target_UNK_CHAR_SEQ.cuda() if self.use_gpu else Target_UNK_CHAR_SEQ
                _,Target_UNK_CHAR_SEQ = preprocess(Target_UNK_CHAR_SEQ,self.Char_pad_id,self.Char_sos_id,self.Char_eos_id)
                

                ### made from (Bs,seq_len)
                #-------------------
                ###repeat the embeding vector equal to length of char string
                ##embedding that corrspods to the labels 
                ####i suspect this way of geting embeding probanbly i should be usding the preembedding  ci_pl_si, let us see
                yi              = self.Wembedding(greedy_label)
                if self.speller_input == 'emb':
                    Unk_emb_vec = yi
                elif self.speller_input == 'emb_ci_si':
                    Unk_emb_vec     = torch.cat([yi,ci,si],dim=2)
                elif self.speller_input == 'emb_ci':
                    Unk_emb_vec     = torch.cat([yi,ci],dim=2)
                elif self.speller_input == 'emb_si':
                    Unk_emb_vec     = torch.cat([yi,si],dim=2)
                else:
                    print("unknown speller input")
                    exit(0)
#                Unk_emb_vec     = torch.cat([yi,ci,si],dim=2) if self.use_si_ci_spelling else yi    ####output of word lstm

                Unk_emb_vec     = Unk_emb_vec.cuda() if self.use_gpu else Unk_emb_vec
                yi_c = torch.repeat_interleave(Unk_emb_vec, Target_UNK_CHAR_SEQ.size(1), dim=0)
                ####Spelling Attention or no attention speller code 
                char_yout , alpha_i, si_out_char,ci_out_char, hn1_out, cn1_out = self.Spelling_step(H, yi_c, hn1, cn1, si, ci, alpha_i_prev)
                char_yout    = torch.transpose(char_yout, 0, 1)
                
                ##### if you are connection Word lstm to char lstm and char lstm to word lstms 
                ## and updating only the the unk words in batch  then you neecd to modify hn1,cn1 such that only the elements with unk gets changed and the rest are intact
                ###i find a utt_with_unk 
                #============================================================================
                if self.spell_everything:
                    utt_with_unk = (greedy_label == greedy_label)
                    ##All Ture
                else:
                        ##Some Ture
                    utt_with_unk = (greedy_label > self.mask_id)
                        ##only makes sence to change when all the unk labels are spelled
                if self.cond_word_on_speller:
                    utt_with_unk_vec   = utt_with_unk.unsqueeze(2).expand_as(hn1)*1
                    hn1_out = hn1 * (1-utt_with_unk_vec) + (utt_with_unk_vec) * hn1_out
                    cn1_out = cn1 * (1-utt_with_unk_vec) + (utt_with_unk_vec) * cn1_out
                #============================================================================
                #breakpoint()
                char_yout_flatten    = char_yout.contiguous().view(-1,self.Char_out_size)
                Target_UNK_CHAR_SEQ_flatten = Target_UNK_CHAR_SEQ.contiguous().view(-1)
                #====================================
                CHAR_HYP = torch.argmax(F.softmax(char_yout_flatten,dim=1),dim=1)
                Target_HYP = Target_UNK_CHAR_SEQ_flatten.data.cpu().numpy()

                CHAR_HYP = CHAR_HYP.data.cpu().numpy()
                Speller_cer = compute_cer(Target_HYP,CHAR_HYP,self.pad_index)
                #===================================
                ### char_yout(Bs*seq_len) ;;;; Target_UNK_CHAR_SEQ (Bs*seq_len)
                char_lstm_loss_perlabel = char_loss(char_yout_flatten, Target_UNK_CHAR_SEQ_flatten, self.Ch_SIL_tok, normalize_length=True, smoothing=self.char_label_smoothing,Ignore_padding=None)



                ##remap to ### made from (Bs,seq_len)
                char_lstm_loss_perutt = char_lstm_loss_perlabel.view(batch_size,-1).sum(dim=1)
                #======================================================
                if self.word_prior_nomalize_speller:
                    char_lstm_loss_perutt = char_lstm_loss_perutt * self.Word_priors[greedy_label.squeeze()]
                #======================================================
                char_lstm_loss = (char_lstm_loss_perutt[utt_with_unk.squeeze()])
                #
                ###norm 
                if self.norm_spell_loss:
                    #remove_the_padding_spelling = torch.all(Target_UNK_CHAR_SEQ[:,1:]==self.Char_pad_id,dim=1)
                    char_lstm_loss = (char_lstm_loss.sum() / Target_UNK_CHAR_SEQ.shape[0])
                else:
                    char_lstm_loss = char_lstm_loss.sum()

                ########
                return char_lstm_loss, char_yout, alpha_i, si_out_char, ci_out_char, hn1_out, cn1_out, Speller_cer
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------

        def forward(self,H,teacher_force_rate,Char_target,Word_target,L_text,only_speller):
                #breakpoint()
                _,Char_target = preprocess(Char_target,self.Char_pad_id,self.Char_sos_id,self.Char_eos_id)
                _,Word_target = preprocess(Word_target,self.pad_index,self.sos_id,self.eos_id)

                Char_target   = Char_target.cuda() if self.use_gpu else Char_target
                Word_target   = Word_target.cuda() if self.use_gpu else Word_target
                #===================================================================
                ###empty lists to store the label predictions
                output_seq , attention_record, greedy_label_seq  = [], [], []
                decoder_steps           = Word_target.size(1)
                batch_size              = H.size(1)
                AUCO_SEQ_len            = H.size(0)
                cost                    = 0
                Char_cer                = 0
                Speller_cer_list        = []
                #--------------------------------------------------------------------
                ####initiaizing the decoder LSTMs and attentions 
                pred_label, yi, si, hn1, cn1, ci, alpha_i_prev =self.initialize_decoder_states(batch_size,AUCO_SEQ_len)
                #---------------------------------------------------------------------------
                #---------------------------------------------------------------------------
                for d_steps in range(decoder_steps):                        
                        yout, alpha_i_W, si_out_W, hn1_out_W, cn1_out_W, ci_out_W = self.select_step(H, yi, hn1, cn1, si, alpha_i_prev,ci) 
                        pred_out=F.softmax(yout,2)      
                        #---------------------------------------------                     
                        char_lstm_loss = 0
                        ###excluding last step as EOS can not be UNK
                        if self.use_speller and (d_steps <decoder_steps-1):
                        #if 1:
                            greedy_label=torch.argmax(pred_out,2)
                            if (self.spell_everything) or torch.any(greedy_label>self.mask_id): #==self.word_unk):
                                
                                #print(L_text.shape,d_steps,Word_target.shape)
                                UNK_TEXT = L_text[:,d_steps] if '_word.model' in self.Word_model_path else None
                                #UNK_TEXT=None
                                UNK_TEXT_BPE = None if '_word.model' in self.Word_model_path else Word_target[:, d_steps].data.cpu().numpy().tolist()
                                


                                #breakpoint()
                                char_lstm_loss, char_yout, alpha_i_char, si_out_char, ci_out_char, hn1_out_char, cn1_out_char, Speller_cer = self.Spell_Rnn(UNK_TEXT_BPE,UNK_TEXT, H, hn1_out_W, cn1_out_W, si_out_W, alpha_i_W, ci_out_W, greedy_label,yout)
                                Speller_cer_list.append(Speller_cer)

                                ####if some other assignments needed 
                                ##they will condition the lexume on char
                                if self.cond_word_on_speller:
                                    hn1_out_W, cn1_out_W = hn1_out_char, cn1_out_char
                        

                        #-------------------------------------------- 
                        sec_yout = torch.cat([yout[:,:,0].unsqueeze(2), torch.logsumexp(yout[:,:,self.mask_id+1:], dim=2).unsqueeze(2), yout[:,:,2:self.mask_id+1]],dim=2)
                        present_word_cost = cal_loss(sec_yout.squeeze(0),Word_target[:,d_steps],self.pad_index, normalize_length=True, smoothing=self.label_smoothing)
                        cost += (1-self.speller_cost_weight)* present_word_cost + self.speller_cost_weight * char_lstm_loss
                        #-----------------------------------------
                        
                        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

                        if teacher_force:
                                pred_label=Word_target[:,d_steps]
                                utt_with_unk = (pred_label > self.mask_id)
                                max_oov = torch.argmax(F.softmax(yout[:,:,self.mask_id+1:],2),2).squeeze(0) + self.mask_id+1
                                pred_label = torch.mul(pred_label,~utt_with_unk) + torch.mul(max_oov,utt_with_unk)
                        else:
                                pred_label=torch.argmax(pred_out,2)
                                pred_label=pred_label.squeeze(0)
                        #-----------------------------------------

                        yi=self.Wembedding(pred_label).unsqueeze(0)
                        hn1, cn1, si, ci, alpha_i_prev =  hn1_out_W, cn1_out_W, si_out_W, ci_out_W, alpha_i_W 
                        output_seq.append(pred_out)
                        attention_record.append(alpha_i_W)
                ###=====================================================================
                ###=====================================================================
                #Computing WER
                #import pdb; pdb.set_trace()

                attention_record = torch.cat(attention_record,dim=1)
                output_seq = torch.cat(output_seq,dim=0).transpose(1,0)
                output_seq = torch.argmax(output_seq,2)
                
                #####
                ####The transformer type error measure does not correspond between training and testing
                #other two error does not corrspond with the error of the systems
                OP=output_seq.data.cpu().numpy(); OP1=np.asarray(OP.flatten())
                WP=Word_target.data.cpu().numpy(); WP1=np.asarray(WP.flatten())
                Word_cer=compute_cer(WP1,OP1,self.pad_index)

                ###======================================================================
                #import pdb; pdb.set_trace()
                Char_CTC_loss=0
                # if self.compute_ctc:
                #     ctc_output = self.CTC_output_layer(H)
                #     #ctc_output = ctc_output.transpose(0,1)
                #     log_probs=torch.nn.functional.log_softmax(ctc_output,dim=2)
                #     input_lengths  = torch.IntTensor([H.size(0)],).repeat(H.size(1))
                #     target_lengths = torch.IntTensor([Char_target.size(1),]).repeat(Char_target.size(0))
                #     input_lengths  =   Variable(input_lengths, requires_grad=False).contiguous()
                #     target_lengths =  Variable(target_lengths, requires_grad=False).contiguous()
                #     input_lengths   = input_lengths.cuda() if self.use_gpu else input_lengths
                #     target_lengths  = target_lengths.cuda() if self.use_gpu else target_lengths
                #     Char_CTC_loss   = self.CTC_Loss(log_probs,Char_target,input_lengths,target_lengths)
                #     CTC_norm_factor = H.size(0)

                #     #### check the normalizations ##mean acorss batch and normalize per-frame
                #     Char_CTC_loss = Char_CTC_loss.mean()/CTC_norm_factor
                #     ###======================================================================
                #     cost = Char_CTC_loss * self.ctc_weight + cost * (1-self.ctc_weight)
                ###=====================================================================
                ###=====================================================================

                Char_cer = mean(Speller_cer_list) if Speller_cer_list else 1
                Decoutput_dict={'cost':cost,
                                'output_seq':output_seq,
                                'attention_record':attention_record,
                                'Char_cer':Char_cer,
                                'Word_cer':Word_cer}
                return Decoutput_dict
        #---------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------
#======================================================================================================================
        def decode_with_beam_LM(self,H,LM_model,Am_weight,beam,gamma,len_pen,ins_pen=0,lm_coupling=0):
                #breakpoint()
                batch_size = H.size(1)
                AUCO_SEQ_len = H.size(0)
                output_seq , attention_record, greedy_label_seq = [], [], []
                #---------------------------------------------------------------------------------------
                if batch_size !=1:
                        print("does not support batch_size greater than 1")
                        exit()
                #---------------------------------------------------------------------------------------
                max_len=int(AUCO_SEQ_len*len_pen)+1
                
                pred_label, yi, si, hn1, cn1, ci, alpha_i_prev = self.initialize_decoder_states(batch_size,AUCO_SEQ_len)
                #---------------------------------------------------------------------------
                
                if Am_weight < 1:
                    rnnlm_h0,rnnlm_c0=LM_model.Initialize_hidden_states(batch_size)
                else:
                     rnnlm_h0, rnnlm_c0 = 0, 0
                     rnnlm_h0_out, rnnlm_c0_out = 0, 0

                #---------------------------------------------------------------------------                
                ys = torch.ones(1,batch_size).fill_(self.sos_id).type_as(H).long()

                hyp = {'score': 0.0, 'yseq': ys,'state': [hn1,cn1,si,alpha_i_prev,ci,rnnlm_h0,rnnlm_c0],'alpha_i_list':alpha_i_prev,'yseq_sym':[['sos_id']]}
                hyps = [hyp]
                ended_hyps = []

                for d_steps in range(max_len):
                        hyps_best_kept=[]
                        for hyp in hyps:
                                ys=hyp['yseq']

                                #--------------------------- 
                                pred_label = pred_label if d_steps==0 else ys[:,-1].unsqueeze(1)
                                yi=self.Wembedding(pred_label)
                                #-------------------------------
                                ####no_teacher forcing so always predicted label

                                present_state=hyp['state']
                                [hn1,cn1,si,alpha_i_prev,ci,rnnlm_h0,rnnlm_c0]=present_state
                                
                                yout, alpha_i_W, si_out_W, hn1_out_W, cn1_out_W, ci_out_W = self.select_step(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                                Am_pred = F.log_softmax(yout,2)

                                #UNK_TEXT, H, hn1_out_W, cn1_out_W, si_out_W, alpha_i_W, ci_out_W, greedy_label)
                                #Am_pred = Am_pred #-1.0 #####insertion_penalty 
                                
                                if Am_weight < 1:    
                                    #breakpoint()
                                    ys_lm_input = ys[:,-1].unsqueeze(0) if d_steps>0 else ys
                                    lm_predict_out,(rnnlm_h0_out, rnnlm_c0_out)=LM_model.predict_rnnlm(ys_lm_input,h0=rnnlm_h0, c0=rnnlm_c0)
                                    Lm_pred = F.log_softmax(lm_predict_out,2)
                                    pred_out = Am_weight*Am_pred + (1-Am_weight)*Lm_pred
                                else:
                                    pred_out=Am_pred

                                ####set new states after am and LM predictions
                                new_state= [hn1_out_W, cn1_out_W, si_out_W, alpha_i_W, ci_out_W, rnnlm_h0_out,rnnlm_c0_out]
                                #hyp['state']=new_state

                                #--------------------------------------
                                #beam-------code
                                local_best_scores, local_best_ids = torch.topk(pred_out, beam, dim=2)
                                #print(local_best_scores.size(),local_best_ids.size())
                                #Eos threshold
                                # pdb.set_trace()
                                ####------------------------------------------------------------------
                                #breakpoint()
                                #print(ys)
                                EOS_mask=local_best_ids==self.eos_id
                                if (EOS_mask.any()) and beam>1:
                                    KEEP_EOS=local_best_scores[EOS_mask] > gamma * torch.max(local_best_scores[~EOS_mask])
                                    #print(KEEP_EOS)
                                    if (KEEP_EOS.item()):
                                        pass;
                                    else:
                                        local_best_scores[EOS_mask]=-1000
                                #print(local_best_scores,local_best_ids)
                                ####------------------------------------------------------------------                             
                                ####------------------------------------------------------------------
                                all_candidates=[]
                                for j in range(beam):
                                        new_hyp = {}
                                        #breakpoint()
                                        present_lab=torch.tensor([[int(local_best_ids[0,0,j])]])
                                        new_hyp['score'] = hyp['score'] + local_best_scores[0,0,j]
                                        present_lab=present_lab.cuda() if H.is_cuda else present_lab 
                                        new_hyp['yseq'] = torch.cat((ys,present_lab),dim=1)
                                        present_id=int(local_best_ids[0,0,j])
                                        #
                                        #####only when speller is used
                                        #==================================================
 #                                       if self.use_speller:
                                        if (present_id < self.targets_no): # and not (present_id > self.mask.id): # self.word_unk):
                                              word_new_symbol = self.Word_model.DecodeIds([present_id])

                                              #breakpoint()
#                                              char_beam = self.predict_from_speller(H,hn1_out_W, cn1_out_W, si_out_W, alpha_i_W, ci_out_W, present_lab)
 #                                             new_symbol = char_beam[0].get('Text_seq')
                                              new_symbol = [[word_new_symbol]]

                                              #print(new_symbol,word_new_symbol, present_id)
                                        elif (present_id == self.pad_index):
                                                new_symbol = [['pad_id']]
                                        elif (present_id == self.sos_id):
                                                new_symbol = [['sos_id']]
                                        elif (present_id == self.eos_id):
                                                new_symbol = [['eos_id']]
                                          ##==================================================
                                        else: # present_id == self.word_unk:
                                            if self.use_speller:
                                              char_beam = self.predict_from_speller(H, hn1_out_W, cn1_out_W, si_out_W, alpha_i_W, ci_out_W, present_lab)
                             
                                              new_symbol = "<UNK"+str(present_id)+">"+str(char_beam[0].get('Text_seq'))
                                            else:
                                              new_symbol = "<UNK"+str(present_id)+">"
                                            new_symbol = [[new_symbol]]
#                                        else:
 #                                            pass;
#                                        else:
#                                          if (present_id < self.targets_no):
                                            
                                        #==================================================
                                        #breakpoint()
                                        #print(present_id,present_lab,new_symbol,hyp['yseq_sym'])
                                        new_hyp['yseq_sym'] =  hyp['yseq_sym'] + new_symbol
                                        #==================================================

                                        new_hyp['state'] = new_state
                                        new_hyp['alpha_i_list'] = torch.cat((hyp['alpha_i_list'],new_state[3]),dim=1)

                                        ####call_speller
                                        hyps_best_kept.append(new_hyp)
                                        #print(new_hyp['yseq'],new_hyp['score'])
                                #========================================
                                hyps_best_kept = sorted(hyps_best_kept,key=lambda x: x['score'],reverse=True)[:beam]
#                                print(hyps_best_kept['yseq'],hyps_best_kept['score'])
                                #===============================================
                                #print(hyps_best_kept)
                        #===============================================
                        ####remving the hypes with eos with out further computations
                        remained_hyps = []
                        for hyp in hyps_best_kept:
                                hyp_len=hyp['yseq'][0].size()[0]
                    #            print(hyp['yseq'])
                                if hyp['yseq'][0, -1].item() == self.eos_id and d_steps>0:
                                        ended_hyps.append(hyp)
                                else:
                                        remained_hyps.append(hyp)
                        hyps = remained_hyps
                        ##-------------------------------------
                #-------------------------------------
                        #print(d_steps,len(ended_hyps))
                        # if len(ended_hyps) > beam:
                        #     break;

                ### add the unfinishe hyps to finished hypes at the end of max_len
                if len(ended_hyps)==0:
                    ended_hyps=remained_hyps
                
                
                ###add the left over hyps
                ended_hyps += hyps
                ### sort the the hypds bases on score
                nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), beam)]
                print("nbest hyps size",len(nbest_hyps))
                #-----------------------
                ### Convert the hyps to text and add the text seq to dict
                output_dict=[]
                for OP in nbest_hyps:
                #    print(OP['state'])
                    OP.pop('state')
                 #   print(OP['yseq'])
                    unks = torch.Tensor([[(x > self.mask_id)  for x in OP['yseq'][0] ]]).bool()
                  #  print(unks)
                    replace = torch.ones(OP['yseq'].size(),dtype=torch.long)
                  #  print(replace)
                    yseq=OP['yseq'].where(~unks, replace)
                  #  print(yseq)
                    #OP['Text_seq']=self.get_charecters_for_sequences(OP['yseq'])
                    OP['Text_seq']=self.get_charecters_for_sequences(yseq,self.Word_model, self.pad_index, self.eos_id,self.word_unk) #OP['yseq'],self.Word_model, self.pad_index, self.eos_id,self.word_unk)
                    output_dict.append(OP)
                #-------------------------------------                
                #-------------------------------------                                                 
                return output_dict

                #=======================================================================================
        #=======================================================================================
        #--------------------------------------------
        # def get_charecters_for_sequences(self,input_tensor):
        #     """ Takes pytorch tensors as in put and print the text charecters as ouput,  
        #     replaces sos and eos as unknown symbols and ?? and later deletes them from the output string"""
        #     output_text_seq=[]
        #     final_token_seq=input_tensor.data.numpy()
        #     final_token_seq=np.where(final_token_seq>self.pad_index,self.Word_SIL_tok,final_token_seq)
        #     text_sym_sil_tok=self.Word_model.DecodeIds([self.Word_SIL_tok])

        #     for i in final_token_seq:
        #         i=i.astype(np.int).tolist()
        #         text_as_string=self.Word_model.DecodeIds(i)
        #         text_as_string=text_as_string.replace(text_sym_sil_tok,"")
        #         output_text_seq.append(text_as_string)
        #     return output_text_seq
        #=======================================================================================
        def get_charecters_for_sequences(self,input_tensor,Word_model,pad_index,eos_id,word_unk):
            ##made it little geniric may be required for future i dont know...?

            """ Takes pytorch tensors as in put and print the text charecters as ouput,  
            replaces sos and eos as unknown symbols and ?? and later deletes them from the output string"""

            output_text_seq=[]
            final_token_seq=input_tensor.data.numpy()          
            text_sym_sil_tok=Word_model.DecodeIds([word_unk])

            for i in final_token_seq:
                ##delete the elements if the model produces good labels after the eos they need to be removed
                i=i.astype(np.int).tolist()
                utt_eos_index = i.index(eos_id) if eos_id in i else None

                #--------------------------------------
                ####if eos exists in the utterance remove the seq after eos
                if utt_eos_index:
                    i=i[:utt_eos_index]

                ##if no eos in the seq then take full seq
                i=[x for x in i if x<pad_index]
                #--------------------------------------

                text_as_string = Word_model.DecodeIds(i)
                text_as_string = text_as_string.replace(text_sym_sil_tok," <UNK>")
                output_text_seq.append(text_as_string)
            return output_text_seq
        #=======================================================================================
        def initialize_decoder_states(self,batch_size,AUCO_SEQ_len):
                pred_label = torch.ones(1,batch_size).fill_(self.sos_id).long()
                pred_label = pred_label.cuda() if self.use_gpu else pred_label

                yi  = self.Wembedding(pred_label)
                si  = self.init_Hidden(batch_size) ######si=torch.mean(H,0,keepdim=True) could also be used
                hn1 = self.init_Hidden(batch_size)
                cn1 = self.init_Hidden(batch_size)


                hn1 = torch.cat([hn1]*self.Lexume_layers,dim=0)
                cn1 = torch.cat([cn1]*self.Lexume_layers,dim=0)



                ci  = self.init_Hidden(batch_size)
                alpha_i_prev = self.init_LOC_Att_vec(batch_size,AUCO_SEQ_len)
                return pred_label, yi, si, hn1, cn1, ci, alpha_i_prev
        #--------------------------------------------
        #--------------------------------------------
        def init_Hidden(self,batch_size):
                result = Variable(torch.zeros(1,batch_size,self.hidden_size))
                result=result.cuda() if self.use_gpu else result
                return result
        #--------------------------------------------
        def init_Output(self,batch_size):
                result = Variable(torch.zeros(1,batch_size,self.hidden_size))
                result=result.cuda() if self.use_gpu else result
                return result
        #--------------------------------------------
        def init_Att_vec(self):
                result = Variable(torch.zeros(1,1,self.hidden_size))
                result=result.cuda() if self.use_gpu else result
                return result
       #--------------------------------------------
        def init_LOC_Att_vec(self,B,n):
                result = Variable(torch.zeros(n,1,B))
                result=result.cuda() if self.use_gpu else result
                return result
        #--------------------------------------------
        def init_embeding_vector(self):
                result = Variable(torch.zeros(1,1,self.emb_dim))
                result=result.cuda() if self.use_gpu else result
                return result
        #-------------------------------------------
        def Non_Autoreg_beamsrearch(self,Posterior_matrix,Tok_model,Tok_padding_id,Tok_sos_id,Tok_eos_id,Tok_unk_id):

                beam=5
                gamma=1
                max_len=Posterior_matrix.size(1)
                batch_size=Posterior_matrix.size(0)
                ys=torch.ones(1,batch_size).fill_(Tok_sos_id).long()

                hyp = {'score': 0.0, 'yseq': ys,'yseq_sym':['sos_id']}
                hyps = [hyp]
                ended_hyps = []
                for d_steps in range(max_len):
                        hyps_best_kept=[]
                        for hyp in hyps:
                                ys=hyp['yseq']
                                #--------------------------- 
                                
                                yout=Posterior_matrix[:,d_steps,:]
                                yout=yout.unsqueeze(0)
                                Am_pred = F.log_softmax(yout,2)                              
                                pred_out=Am_pred

                                #--------------------------------------
                                local_best_scores, local_best_ids = torch.topk(pred_out, beam, dim=2)

                                #Eos threshold
                                ####------------------------------------------------------------------
                                EOS_mask=local_best_ids==Tok_eos_id
                                if (EOS_mask.any()) and beam>1:
                                    KEEP_EOS=local_best_scores[EOS_mask] > gamma * torch.max(local_best_scores[~EOS_mask])
                                    #print(KEEP_EOS)
                                    if (KEEP_EOS.item()):
                                        pass;
                                    else:
                                        local_best_scores[EOS_mask]=-1000
                                ####------------------------------------------------------------------                             
                                ####------------------------------------------------------------------
                                all_candidates=[]
                                for j in range(beam):
                                        new_hyp = {}

                                        present_lab=torch.tensor([[int(local_best_ids[0,0,j])]])
                                        new_hyp['score'] = hyp['score'] + local_best_scores[0,0,j]
                                        present_lab=present_lab.cuda() if Posterior_matrix.is_cuda else present_lab 
                                        new_hyp['yseq'] = torch.cat((ys,present_lab),dim=1)

                                        present_id=int(local_best_ids[0,0,j])
                                        hyps_best_kept.append(new_hyp)

                                #========================================
                                hyps_best_kept = sorted(hyps_best_kept,key=lambda x: x['score'],reverse=True)[:beam]
                                #===============================================
                        #
                        #===============================================
                        remained_hyps = []
                        for hyp in hyps_best_kept:
                                hyp_len=hyp['yseq'][0].size()[0]
                                if hyp['yseq'][0, -1].item() == self.eos_id and d_steps>0:
                                        ended_hyps.append(hyp)
                                else:
                                        remained_hyps.append(hyp)

                        hyps = remained_hyps
                        

                #===============================================
                ### add the unfinishe hyps to finished hypes at the end of max_len
                if len(ended_hyps)==0:
                        ended_hyps=remained_hyps

                #===============================================
                ### sort the the hypds bases on score
                nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), beam)]

                #-----------------------
                #===============================================
                ### Convert the hyps to text and add the text seq to dict
                #breakpoint()
                output_dict=[]
                for OP in nbest_hyps:
                    OP['Text_seq']=self.get_charecters_for_sequences(OP['yseq'],Tok_model,Tok_padding_id,Tok_eos_id,Tok_model.unk_id())
                    output_dict.append(OP)
                #-------------------------------------
                #-------------------------------------                                                 
                return output_dict

#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
# ##debugger
# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/')
# import Attention_arg
# from Attention_arg import parser
# args = parser.parse_args()
# print(args)


# #import pdb;pdb.set_trace()
# # Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__word.model'
# # Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__char.model'
# # text_file = '/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/normalized_text_full_train_text_100lines'

# args.Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'

# H = np.load("H.npy",allow_pickle=True)      #torch.randint(low=0, high=1000,size=(10,6))
# H = torch.Tensor(H)
# H = Variable(H, requires_grad=False)

# #torch.rand((81,10,320))
# #import pdb;pdb.set_trace()

# # Word_target=torch.randint(low=0, high=1000,size=(10,6))
# # Char_target=torch.randint(low=0, high=20,size=(10,25))

# Word_target = np.load("smp_label.npy")      #torch.randint(low=0, high=1000,size=(10,6))
# Word_target = torch.IntTensor(Word_target)
# Word_target = Variable(Word_target, requires_grad=False).contiguous().long()


# Char_target=np.load("smp_word_label.npy")  #torch.randint(low=0, high=20,size=(10,25))
# Char_target = torch.IntTensor(Char_target)
# Char_target = Variable(Char_target, requires_grad=False).contiguous().long()


# text_dict = {line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(text_file)}
# text_trans_list = [text_dict.get(T) for T in text_dict.keys()]
# text_trans_list_length = [len(text_dict.get(T)) for T in text_dict.keys()]
# L_text = pad_sequences(text_trans_list,maxlen=max(text_trans_list_length),dtype=object,padding='post',value='unk')
# #L_text = L_text[:10] 
# L_text=np.load("smp_trans_text.npy",allow_pickle=True)


# # import sentencepiece as spm
# # Word_model = spm.SentencePieceProcessor()
# # Char_model = spm.SentencePieceProcessor()
# # Word_model.Load(join(Word_model_path))
# # Char_model.Load(join(Char_model_path))
# #import pdb;pdb.set_trace()

# hidden_size = 320;
# compute_ctc = True
# ctc_weight  = 0.5
# use_gpu = 0

# use_speller = False
# use_word = True
# ctc_target_type='word'
# teacher_force_rate=0.6
# spell_loss_perbatch=False
# label_smoothing=0.1
# #import pdb;pdb.set_trace()

# args.attention_type='LAS'
# Dec=decoder(args)
# print(Dec)

# #decoder_weight="/mnt/matylda3/vydana/HOW2_EXP/Timit/models/Timit_Conv_Res_LSTM_3layers_256_LSTMSS_ls0.1/decoder_model_epoch_49_sample_17151_7.7726516959396275___1205.532112121582__0.4075471698113208"
# #Dec.load_state_dict(torch.load(decoder_weight, map_location=lambda storage, loc: storage),strict=True)

# H = H.cuda() if use_gpu else H

# Dec(H,teacher_force_rate,Char_target,Word_target,L_text)
# LM_model=None
# Am_weight=1
# beam=5
# gamma=1
# len_pen=1
# #print(H[:,0,:].unsqueeze(1).shape)
# #exit(0)
# #
# #BEAM_OUTPUT=Dec.decode_with_beam_LM(H[:,0,:].unsqueeze(1),LM_model,Am_weight,beam,gamma,len_pen)
# #print(BEAM_OUTPUT)
# print('-------Over---------')



