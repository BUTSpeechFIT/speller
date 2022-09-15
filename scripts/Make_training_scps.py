#! /usr/bin/python

#*******************************
import sys
import os
from os.path import join, isdir
from random import shuffle
import glob

sys.path.insert(0,'/mnt/matylda6/iegorova/speller/scripts') #'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from Load_sp_model import Load_sp_models
from Make_ASR_scp_text_format_fast import format_tokenize_data

import Attention_arg
from Attention_arg import parser
args = parser.parse_args()


if not isdir(args.data_dir):
        os.makedirs(args.data_dir)

format_tokenize_data(scp_file=glob.glob(args.train_path + "*"),transcript=args.text_file,Translation=args.text_file,outfile=open(join(args.data_dir,'train_scp'),'w'),Word_model_path=args.Word_model_path,Char_model_path=args.Char_model_path)
format_tokenize_data(scp_file=glob.glob(args.dev_path + "*"),transcript=args.text_file,Translation=args.text_file,outfile=open(join(args.data_dir,'dev_scp'),'w'), Word_model_path=args.Word_model_path,Char_model_path=args.Char_model_path)








