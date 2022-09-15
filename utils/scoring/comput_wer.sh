#!/usr/bin/bash

#. ./path.sh

scrng_pth=$3
GT=$1
hyp=$2


align-text ark,t:$GT ark,t:$hyp ark,t:$scrng_pth/alignment.txt_val
cat alignment.txt_val|$scrng_pth/wer_per_utt_details.pl>$scrng_pth/per_utt_alignment.txt_val
compute-wer --text --mode=present ark:$GT  ark,t:$hyp 2>&1> wer_val
