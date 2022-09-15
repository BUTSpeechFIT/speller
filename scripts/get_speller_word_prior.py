#!/usr/bin/python


from collections import Counter
import numpy as np

def Get_Priors(Wout_size,files):
        #
        #file_path="LIBRISP960hrs_training_Data_249_scps/train_scp"
        prior_mat = np.zeros((1,Wout_size),dtype=np.float)
        #
        prior_dict={}
        for i in range(0,Wout_size):
                prior_dict[i] = 0
        #
        #
        C=Counter()
        C.update(prior_dict)
        for inp_file in files:
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
                                C.update(word_tokens)
        #====================================================
        for key,value in C.items():
                prior_mat[0,key] = value
        #
        return prior_mat
        #====================================================
