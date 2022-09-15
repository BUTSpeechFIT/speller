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
#sys.path.insert(0, '/mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/Basic_Attention_V1/') #'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE')
import batch_generators.CMVN
from batch_generators.CMVN import CMVN
from Load_sp_model import Load_sp_models


#===============================================
#-----------------------------------------------  
class DataLoader(object):

    def __init__(self,files, max_batch_label_len, max_batch_len, max_feat_len, max_label_len, Word_model, Char_model, queue_size=100,apply_cmvn=1):

        self.files = files
        if self.files==[]:
                print('input to data generator in empty')
                exit(0)


        self.text_file_dict ={} 

        self.Word_model = Word_model
        self.Char_model = Char_model
        self.max_batch_len = max_batch_len
        self.max_batch_label_len = max_batch_label_len
        self.max_feat_len = max_feat_len
        self.max_label_len = max_label_len
        self.apply_cmvn = apply_cmvn


        self.queue = queue.Queue(queue_size)
        self.Word_padding_id = self.Word_model.__len__()
        self.Char_padding_id = self.Char_model.__len__()
        self.word_space_token   = self.Word_model.EncodeAsIds('_____')[0]
        
    
        self._thread = Thread(target=self.__load_data)
        self._thread.daemon = True
        self._thread.start()

    
    def __reset_the_data_holders(self):
        self.batch_data=[]
        self.batch_labels=[]
        self.batch_names=[]
        self.batch_length=[]
        self.batch_label_length=[]
        
        self.batch_word_labels=[]
        self.batch_word_label_length=[]
        
        self.batch_word_text=[]
        self.batch_word_text_length=[]

        self.batch_word_text_tgt=[]
        self.batch_word_text_length_tgt=[]
    
    #---------------------------------------------------------------------
    def make_batching_dict(self):
        #----------------------------------------
        smp_feat=pad_sequences(self.batch_data,maxlen=max(self.batch_length),dtype='float32',padding='post',value=0.0)
        smp_char_labels=pad_sequences(self.batch_labels,maxlen=max(self.batch_label_length),dtype='int32',padding='post',value=self.Char_padding_id) 
        smp_word_label=pad_sequences(self.batch_word_labels,maxlen=max(self.batch_word_label_length),dtype='int32',padding='post',value=self.Word_padding_id)
        smp_trans_text=pad_sequences(self.batch_word_text, maxlen=max(self.batch_word_text_length),dtype=object,padding='post',value=' ')
        smp_trans_text_tgt=pad_sequences(self.batch_word_text_tgt, maxlen=max(self.batch_word_text_length_tgt),dtype=object,padding='post',value=' ')

        batch_data_dict={
            'smp_names':self.batch_names,
            'smp_feat':smp_feat,
            'smp_char_label':smp_char_labels,
            'smp_word_label':smp_word_label,
            'smp_trans_text':smp_trans_text,
            'smp_trans_text_tgt': smp_trans_text_tgt,
            'smp_feat_length':self.batch_length,
            'smp_label_length':self.batch_label_length,
            'smp_word_label_length':self.batch_word_label_length,
            'smp_word_text_length':self.batch_word_text_length,
            'smp_word_text_length_tgt':self.batch_word_text_length_tgt}
        return batch_data_dict
    #------------------------------------------
    #------------------------------------------
    def __load_data(self):
        ###initilize the lists
        while True:
            self.__reset_the_data_holders()
            max_batch_label_len = self.max_batch_label_len
            random.shuffle(self.files)
            for inp_file in self.files:
                #print(self.files)
                with open(inp_file) as f:
                    for line in f:
                        #============================
                        split_lines=line.split(' @@@@ ')
                        #============================
                        ##assigining
                        key = split_lines[0]
                        scp_path = split_lines[1]
                        #============================
                        ### Char labels
                        #============================
                        src_text = split_lines[3] 
                        src_tok = split_lines[4] 
                        src_tok = [int(i) for i in src_tok.split(' ')]  
                        #============================
                        ##Word models
                        #============================
                        tgt_text = split_lines[5]
                        tgt_tok = split_lines[6]
                        tgt_tok = [int(i) for i in tgt_tok.split(' ')]  
                        #============================
                        ### text 
                        #============================
                        char_tokens = src_tok
                        word_tokens = tgt_tok

                        char_labels = src_text.split(' ')
                        word_labels = tgt_text.split(' ')
                        #--------------------------
                        if not (scp_path == 'None'):
                            mat = kaldi_io.read_mat(scp_path)
                            if self.apply_cmvn:
                                mat = CMVN(mat)
                        else:
                            mat=np.zeros((100,249),dtype=np.float32)
                        #--------------------------


                        if (mat.shape[0]>self.max_feat_len) or (mat.shape[0]<len(char_labels)) or (len(char_tokens) > self.max_label_len):
                                print("key,mat.shape,char_labels,char_tokens,self.max_label_len",key,mat.shape,len(char_labels),len(char_tokens),self.max_label_len)
                                continue;

                        #==============================================================
                        ###Add to the list
                        ####
                        self.batch_data.append(mat)                
                        self.batch_names.append(key)
                        self.batch_length.append(mat.shape[0])

                        self.batch_labels.append(char_tokens)
                        self.batch_label_length.append(len(char_tokens))
                        
                        self.batch_word_labels.append(word_tokens)
                        self.batch_word_label_length.append(len(word_tokens))

                        self.batch_word_text.append(char_labels)
                        self.batch_word_text_length.append(len(char_labels))

                        self.batch_word_text_tgt.append(word_labels)
                        self.batch_word_text_length_tgt.append(len(word_labels))   
                        #==============================================================

                        #==============================================================
                        # total_labels_in_batch is used to keep track of the length of sequences in a batch, just make sure it does not overflow the gpu
                        ##in general lstm training we are not using this because self.max_batch_len will be around 10-20 and self.max_batch_label_len is usuvally set very high                         
                        expect_len_of_features=max(max(self.batch_length,default=0),mat.shape[0])
                        expect_len_of_labels=max(max(self.batch_label_length,default=0),len(char_tokens))

                        total_labels_in_batch= (expect_len_of_features + expect_len_of_labels)*(len(self.batch_names)+4)

                        ###check if ypu have enough labels output and if you have then push to the queue
                        ###else keep adding them to the lists
                        if total_labels_in_batch > self.max_batch_label_len or len(self.batch_data)==self.max_batch_len:
                                    # #==============================================================
                                    # ####to clumsy -------> for secound level of randomization 
                                    # CCCC=list(zip(batch_data,batch_names,batch_labels,batch_word_labels,batch_word_text,batch_label_length,batch_length,batch_word_label_length,batch_word_text_length))
                                    # random.shuffle(CCCC)
                                    # batch_data,batch_names,batch_labels,batch_word_labels,batch_word_text,batch_label_length,batch_length,batch_word_label_length,batch_word_text_length=zip(*CCCC)
                                    # #==============================================================

                                    batch_data_dict = self.make_batching_dict()
                                    self.queue.put(batch_data_dict)
                                    ###after pushing data to lists reset them
                                    self.__reset_the_data_holders()
            

            if len(self.batch_names)>0:
                ### Collect the left over stuff  as the last batch
                #-----------------------------------------------
                batch_data_dict = self.make_batching_dict()
                self.queue.put(batch_data_dict)

    def next(self, timeout=30000):
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
# Word_model=Load_sp_models(args.Word_model_path)
# Char_model=Load_sp_models(args.Char_model_path)
# train_gen = DataLoader(files=glob.glob(args.train_path + "*"),max_batch_label_len=20000, max_batch_len=4,max_feat_len=2000,max_label_len=200,Word_model=Word_model,Char_model=Char_model,text_file=args.text_file)
# for i in range(10):
#     B1 = train_gen.next()
#     print(B1.keys())
#     #breakpoint()

