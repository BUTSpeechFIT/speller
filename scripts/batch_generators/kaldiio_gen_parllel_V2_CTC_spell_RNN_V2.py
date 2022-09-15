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
from Load_sp_model import Load_sp_models


#===============================================
#-----------------------------------------------  
class DataLoader(object):

    def __init__(self,files,max_batch_label_len, max_batch_len,Word_model,Char_model,queue_size=100):
        self.files = files
        self.text_file_dict = {line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(args.text_file)}
        
        # self.Word_model = Load_sp_models(args.Word_model_path)
        # self.Char_model = Load_sp_models(args.Char_model_path)

        self.Word_model = Word_model
        self.Char_model = Char_model

        self.max_batch_len = max_batch_len
        self.queue = queue.Queue(queue_size)
        self.max_batch_label_len = max_batch_label_len

        self.Word_padding_id = self.Word_model.__len__()
        self.Char_padding_id = self.Char_model.__len__()
        
        self.word_space_token   = self.Word_model.EncodeAsIds('_____')[0]

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
                            word_tokens=self.Word_model.EncodeAsIds(" ".join(labels))            
                            char_tokens=self.Char_model.EncodeAsIds(" ".join(labels))
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
                            batch_data_dict={'smp_names':batch_names, 'smp_feat':padded_feat, 'smp_label':padded_labels, 'smp_word_label':padded_word_labels,'smp_trans_text':padded_trans_text,'smp_feat_length':batch_length ,'smp_label_length':batch_label_length,'smp_word_label_length':batch_word_label_length,'smp_word_text_length':batch_word_text_length}

                            #(padded_feat, padded_labels,padded_word_labels,padded_trans_text,batch_length,batch_label_length,batch_word_label_length,batch_word_text_length)
                            self.queue.put(batch_data_dict)
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



# ###debugger
# args.Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.text_file = '/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# args.train_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/'
# args.dev_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/dev/'

# train_gen = HariDataLoader(files=glob.glob(args.train_path + "*"),args=args)
# for i in range(10):
#     B1 = train_gen.next()
#     print(B1)
#     import pdb;pdb.set_trace()
#     #print("smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length")
#     #smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length = B1
#     #import pdb;pdb.set_trace()
#     #print('smp_batch_no',i,"smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length")
#     #print(smp_feat.shape,smp_label.shape,smp_word_label.shape,smp_trans_text.shape,len(smp_feat_len),len(smp_label_len),len(smp_word_label_length),len(smp_word_text_length))
