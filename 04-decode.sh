#! /bin/sh

PPATH="/mnt/matylda6/iegorova/speller/"
cd "$PPATH"
export PYTHONUNBUFFERED=TRUE

only_scoring='False'
gpu=0
speller='True'

if [ $speller == 'True' ];
then
  PPATH="$PPATH/speller_models/"
else
  PPATH="$PPATH/baseline_models/"
fi

scoring_path='/mnt/matylda6/iegorova/speller/utils/scoring/'
feat_dir='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/feat_lists/'
model_file="960hrs_10000w_100ch_sh_w_LAS_LOC_ci_2_4_800_lrth30_ls0.1_TF0.6_lr0.0001_clipgrad10_subsamp_maxpool_lstmdr0.3_init_xavier_lin_relu_fbank83_cmvn_warmup20Kexpdecay1000Kmin1e6_decdr0.1_newbob_speller_embs_si_only_0.5_spweight" 
model_dir="$PPATH/models/$model_file"
pre_trained_weight="$model_dir/model_epoch_21_sample_19999_45.31844134417772___293.7165788074484__0.08711591748531028" 
ref_text_file='/mnt/matylda6/iegorova/speller/data/all_text'
vocab_file='/mnt/matylda6/iegorova/speller/Librispeech_960hrs_sentencepiece/sp_model_10000_words_100_chars_word.vocab'

weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res

max_jobs=3000 
mem_req_decoding=15G #20G

gamma=1.0
len_pen=0.5
beam=10
Am_weight=1

for test_set in "dev_clean" "dev_other" "test_clean" "test_other" 
do
  echo $test_set
  test_fol="${feat_dir}/${test_set}/"
#  decoding_tag="_decoding_${test_set}" #_beam_${beam}_${D_path}_gamm$gamma""len_pen$len_pen"
  log_path=$model_dir/dec_log_$test_set
  echo "$log_path"

  mkdir -pv "$log_path"
  mkdir -pv "$log_path/scoring"
  mkdir -pv "$model_dir/decoding_files/plots"
#  text_scoring_file='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/all_text_10000'

  if [ $only_scoring != 'True' ]; 
  then
    decode_cmd="/mnt/matylda6/iegorova/speller/scripts/decoding.py"
    echo $decode_cmd
    /mnt/matylda6/iegorova/speller/utils/queue.pl \
         --max-jobs-run $max_jobs \
         -q all.q@@stable,all.q@@blade \
         --mem $mem_req_decoding \
         -l matylda6=0.01,ram_free=$mem_req_decoding,tmp_free=10G \
         JOB=1:$max_jobs \
         -l 'h=!blade063' \
         $log_path/decoding_job.JOB.log \
         python $decode_cmd \
         --gpu $gpu \
         --model_dir $model_dir \
         --Decoding_job_no JOB \
         --beam $beam \
         --gamma $gamma \
         --Am_weight $Am_weight \
         --len_pen $len_pen\
         --dev_path $test_fol \
         --weight_text_file $weight_text_file\
         --Res_text_file $Res_text_file\
         --text_file $ref_text_file \
         --pre_trained_weight $pre_trained_weight 
  fi
# done

#  . /mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/Basic_Attention_V1/path.sh
  if [ $speller == 'True' ];
  then
    /mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/SCORING_HELPER_SPELLER.sh $log_path $ref_text_file $vocab_file
  else
    /mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/SCORING_HELPER.sh $log_path $ref_text_file $vocab_file
  fi


#  else

#    . /mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/Basic_Attention_V1/path.sh
#    . /mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/SCORING_HELPER_OOV.sh $log_path $text_scoring_file

#  fi

done


