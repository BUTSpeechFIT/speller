#!/usr/bin/bash


#------------------------------------------------------------------------
. ./path.sh
. ./cmd.sh
#------------------------------------------------------------------------
#------------------------------------------------------------------------

nj=50
fbank_conf="conf/fbank.conf"
data_pth="/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/" # <-- point to kaldi-style data folder

for data_dir in train_clean_100 train_clean_360 train_other_500 dev_other test_other dev_clean test_clean
do
data_dir=$data_pth/$data_dir
echo "$data_dir"


#------------------------------------------------------------------------
highres_dir="/mnt/scratch/tmp/iegorova/LS_features/Librispeech_fbank249_cmvn" # <-- point to where you have a lot of space for features
#------------------------------------------------------------------------

echo $fbank_conf
scripts/make_fbank_pitch.sh --fbank-config "$fbank_conf" --cmd "$train_cmd" --nj $nj "$data_dir" $highres_dir $highres_dir
scripts/compute_cmvn_stats.sh --fake $data_dir
scripts/kaldi_feature_extract_mfcc_delta.sh --nj $nj --cmd "$train_cmd" $data_dir $highres_dir
#-----------------------------------
done





