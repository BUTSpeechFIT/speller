#!/usr/bin/python

import sys
import os
from os.path import join, isdir
import sentencepiece as spm

#--------------------------
def Load_sp_models(PATH):
        PATH_model = spm.SentencePieceProcessor()
        PATH_model.Load(join(PATH))
        return PATH_model
#--------------------------

