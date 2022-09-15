#!/usr/bin/python
import kaldi_io
import sys
import os
from os.path import join, isdir
from numpy.random import permutation
import itertools
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import queue
from threading  import Thread
import random
import glob


import sys
sys.path.insert(0, '/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE')
import batch_generators.CMVN
from batch_generators.CMVN import CMVN



import sentencepiece as spm
sp = spm.SentencePieceProcessor()


#===============================================
#-----------------------------------------------  
class HariDataLoader(object):

    def __init__(self, files,Word_model,Char_model,max_batch_label_len,max_batch_len,args,queue_size=100):

        self.files = files
        self.text_file_dict = {line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(args.text_file)}
        self.word_model = Word_model
        self.char_model = Char_model

        self.max_batch_len = max_batch_len
        self.queue = queue.Queue(queue_size)
        self.max_batch_label_len = max_batch_label_len

        self.Word_padding_id = self.word_model.__len__()
        self.Char_padding_id = self.char_model.__len__()

        self.word_space_token   = self.word_model.EncodeAsIds('_____')[0]


        self._reading_finished = False
        self.label_list = {}
        self._thread = Thread(target=self.__load_data)
        self._thread.daemon = True
        self._thread.start()
       

    def __load_data(self):

        batch_data=[];
        batch_labels=[];
        batch_names=[];
        batch_length=[]; 
        batch_label_length=[];

        batch_word_labels=[];
        batch_word_label_length=[];
        batch_word_text=[]

        batch_word_text_length=[]
        max_batch_label_len = self.max_batch_label_len

        # import pdb; pdb.set_trace()
        while True:
            #print('Finished whole data: Next iteraton of the data---------->')
            for name in self.files:
                file_path = name
                for key, mat in kaldi_io.read_mat_scp(file_path): 
                        mat=CMVN(mat)


                        labels=self.text_file_dict.get(key,'None')
                        ####tokernizing the text
                        if labels=='None':
                           print("labels doesnot exixst in the text file for the key", key)
                           continue;
                        else: 
                            word_tokens=self.word_model.EncodeAsIds(" ".join(labels))            
                            char_tokens=self.char_model.EncodeAsIds(" ".join(labels))
                            #print(word_tokens,char_tokens)

                        #-----------------####cases when the data should be thrown away
                        if (labels==None) or (mat.shape[1]==40) or (mat.shape[0]>2000) or (mat.shape[0]<len(labels)):
                            continue;

                        total_labels_in_batch=(max(max(batch_length,default=0),mat.shape[0])+max(max(batch_label_length,default=0),len(char_tokens + word_tokens)))*(len(batch_names)+4)
                        #print(total_labels_in_batch,self.max_batch_label_len)
                        if total_labels_in_batch > self.max_batch_label_len or len(batch_data)==self.max_batch_len:
                            
                            # #==============================================================
                            # ####to clumsy ------->
                            # CCCC=list(zip(batch_data,batch_names,batch_labels,batch_word_labels,batch_word_text,batch_label_length,batch_length,batch_word_label_length,batch_word_text_length))
                            # random.shuffle(CCCC)
                            # batch_data,batch_names,batch_labels,batch_word_labels,batch_word_text,batch_label_length,batch_length,batch_word_label_length,batch_word_text_length=zip(*CCCC)
                            # #==============================================================



                            padded_feat=pad_sequences(batch_data,maxlen=max(batch_length),dtype='float32',padding='post',value=0.0)
                            padded_labels=pad_sequences(batch_labels,maxlen=max(batch_label_length),dtype='int32',padding='post',value=self.Char_padding_id) 

                            padded_word_labels=pad_sequences(batch_word_labels,maxlen=max(batch_word_label_length),dtype='int32',padding='post',value=self.Word_padding_id)
                            padded_trans_text=pad_sequences(batch_word_text,maxlen=max(batch_word_text_length),dtype=object,padding='post',value='')

                            #import pdb; pdb.set_trace()
                            self.queue.put((padded_feat, padded_labels,padded_word_labels,padded_trans_text,batch_length,batch_label_length,batch_word_label_length,batch_word_text_length))
                            #=================================
                            batch_data=[]
                            batch_labels=[]
                            batch_names=[]
                            batch_length=[]
                            batch_labels=[]
                            batch_label_length=[]
                            
                            batch_word_labels=[]
                            batch_word_label_length=[]
                    
                            batch_word_text=[]
                            batch_word_text_length=[]

                        #==============================================================
                        batch_labels.append(char_tokens)
                        batch_label_length.append(len(char_tokens))
                        batch_data.append(mat)                
                        batch_names.append(key)
                        #print(key)
                        batch_length.append(mat.shape[0])
                        
                        batch_word_labels.append(word_tokens)
                        batch_word_label_length.append(len(word_tokens))
                        
                        batch_word_text.append(labels)
                        batch_word_text_length.append(len(labels))


        if len(batch_data) > 0:
            #==============================================================
            padded_feat=pad_sequences(batch_data,maxlen=max(batch_length),dtype='float32',padding='post',value=0.0)
            padded_labels=pad_sequences(batch_labels,maxlen=max(batch_label_length),dtype='int32',padding='post',value=self.Char_padding_id)            
            padded_word_labels=pad_sequences(batch_word_labels,maxlen=max(batch_word_label_length),dtype='int32',padding='post',value=self.Word_padding_id)
            padded_trans_text=pad_sequences(batch_word_text,maxlen=max(batch_word_text_length),dtype=object,padding='post',value='')
            self.queue.put((padded_feat, padded_labels,padded_word_labels,padded_trans_text,batch_length,batch_label_length,batch_word_label_length,batch_word_text_length))
        #======================
        self._reading_finished = True

    def next(self, timeout=30000):
        if self._reading_finished and self.queue.empty():
            return None
        return self.queue.get(block=True, timeout=timeout)
#===================================================================




# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/KAT_Attention')
# import Attention_arg
# from Attention_arg import parser
# args = parser.parse_args()
# print(args)


# # Namespace(Char_model_path='', Res_text_file='weight_folder/weight_file_res', Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model', batch_size=10, char_space_token=3, clip_grad_norm=5.0, compute_ctc=True, conv_dropout=0.1, ctc_target_type='word', ctc_weight=0.5, , encoder_dropout=0.1, encoder_layers=1.0, gpu=0, hidden_size=320, in_channels=1, input_size=120, isresidual=1, kernel_size=3, label_smoothing=0.1, learning_rate=0.0003, lstm_dropout=0.3, max_F_bands=80, max_batch_label_len=50000, max_batch_len=20, max_train_examples=23380, max_val_examples=2039, min_F_bands=30, model_dir='models/Default_folder', n_layers=5, nepochs=100, new_bob_decay=0, no_of_Char_tokens=None, no_of_Word_tokens=None, no_of_checkpoints=2, noise_inj_ratio=0.1, out_channels=256, pre_trained_weight='0', spec_aug_flag=1, spell_loss_perbatch=True, square=2, steps=1, stride=2, teacher_force=0.6, text_file='', time_drop_max=4, time_window_max=4, tr_disp=1000, , use_speller=1, use_word=True, val_batch_size=10, validate_interval=5000, verbosity=None, vl_disp=100, weight_noise_flag=0, weight_text_file='weight_folder/weight_file', word_unk_token=0, **{'retrain_the last layer': 'False'})


# ###debugger
# args.Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.text_file = '/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# args.train_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/'
# args.dev_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/dev/'


# import sentencepiece as spm
# Word_model=spm.SentencePieceProcessor()
# Char_model=spm.SentencePieceProcessor()
# Word_model.Load(join(args.Word_model_path))
# Char_model.Load(join(args.Char_model_path))


# #*************************************************************************************************************************
# #=========================================================================================================================
# #*************************************************************************************************************************
# # model_path_name=join(args.model_dir,'model_architecture_')
# # print(args,file=open(model_path_name,'w+'))
# print(args.no_of_Word_tokens,args.no_of_Char_tokens)

# Word_tokens=Word_model.__len__()
# Word_unk=Word_model.unk_id()
# Word_SPtoken=Word_model.EncodeAsIds('_____')[0]

# Word_IGNORE_ID=Word_tokens 
# Word_padding_id=Word_tokens
# Word_sos_id=Word_tokens+1 
# Word_eos_id=Word_tokens+2 
# Word_output_size=Word_tokens+3 


# Char_tokens=Char_model.__len__()
# Char_SPtoken=Char_model.EncodeAsIds('_____')[0]
# Char_unk=Char_model.unk_id()


# Char_IGNORE_ID=Char_tokens 
# Char_padding_id=Char_tokens
# Char_sos_id=Char_tokens+1 
# Char_eos_id=Char_tokens+2 
# Char_output_size=Char_tokens+3 



# train_gen = HariDataLoader(files=glob.glob(args.train_path + "*"), Word_model=Word_model, Char_model=Char_model, max_batch_label_len=args.max_batch_label_len, max_batch_len=args.max_batch_len,args=args)
# for i in range(1000000):
#     B1 = train_gen.next()
#     #print(i)
#     #print("smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length")
#     smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length = B1
#     #import pdb;pdb.set_trace()
#     print('smp_batch_no',i,"smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length")
#     print(smp_feat.shape,smp_label.shape,smp_word_label.shape,smp_trans_text.shape,len(smp_feat_len),len(smp_label_len),len(smp_word_label_length),len(smp_word_text_length))
