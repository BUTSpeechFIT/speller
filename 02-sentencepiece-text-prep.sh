#!/usr/bin/bash

text_file=$1 
text_file_chars=$2
no_of_word_tokens=$3
no_of_char_tokens=$4
model_path=$5 
#for example:
#./02-sentencepiece-text-prep.sh ../e2e_hari_v3_speller/data/all_text ../e2e_hari_v3_speller/data/all_text 10000 100 Librispeech_960hrs_sentencepiece Librispeech_960hrs_sentencepiece

mkdir $model_path

#-----------
cut -d " " -f1 $text_file> $model_path/utt_id
#-----------
cut -d " " -f2- $text_file|sed 's/  */ /g'> $model_path/utt_text_for_words
cut -d " " -f2- $text_file_chars|sed 's/  */ /g'> $model_path/utt_text_for_chars
#-----------
#Special_string="" 

echo $model_path


#no_of_word_tokens=27000
#no_of_char_tokens=100
#mkdir -pv $model_path

python scripts/sentencepiece_training.py $model_path/utt_text_for_words $model_path/utt_text_for_chars $model_path $no_of_word_tokens $no_of_char_tokens






