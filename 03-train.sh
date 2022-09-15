#! /bin/sh

#$ -q long.q@supergpu*,long.q@facegpu*,long.q@pcspeech-gpu,long.q@pcgpu*
#$ -l gpu=1,mem_free=14G,ram_free=14G,gpu_ram=14G,matylda6=1
#$ -N speller

#$ -o /mnt/matylda6/iegorova/speller/outputs/out_example.txt
#$ -e /mnt/matylda6/iegorova/speller/outputs/out_example.txt

PPATH="/mnt/matylda6/iegorova/speller/models"
cd "$PPATH"
export PYTHONUNBUFFERED=TRUE

###bash variables
only_scoring='False'
scoring_path='/mnt/matylda6/iegorova/speller/utils/scoring/'
stage=2
#------------------------
gpu=1
max_batch_len=40
max_batch_label_len=30000
max_feat_len=6000
max_label_len=1000

tr_disp=1000
vl_disp=40
validate_interval=40000
max_val_examples=5567

compute_ctc=0
ctc_weight=0

learning_rate=0.0005
sp_learning_rate=0.0005
early_stopping=1
clip_grad_norm=10

hidden_size=800
input_size=83
encoder_layers=4
lstm_dropout=0.3
dec_dropout=0.2

kernel_size=3 
stride=2
in_channels=1
out_channels=512
conv_dropout=0.1

warmup_steps=30000
sp_warmup_steps=$(($warmup_steps/$tr_disp))
echo $sp_warmup_steps
init_lr=0.00005

pretrain_encoder=0
pret_enc_lay_init=1
add_layer_after_steps=1

isresidual=1
label_smoothing=0
##adam|Linear_warmup_adam
optimizer='double_linear_warmup_adam' #Linear_warmup_adam'
lstm_proj_act='linear'
apply_cmvn=1

optimizer_style='newbob'
half_period=20000
min_lr=0.000001
start_decay=0
decoder_layers=2
Dec_model='speller_mult_oovs' 
######
###speller options############################################

use_speller=1
speller_input='emb_ci_si'
cond_word_on_speller=0
char_label_smoothing=0
spell_everything=0
speller_cost_weight=0.5
norm_spell_loss=1
reinitialize_speller_always=1
freeze_all_but_speller=1
unfreeze_emb=1
num_OOV_embs=1

###############################################################
#Attention type: LAS|Collin_monotonc|Location_aware|LAS_LOC|LAS_LOC_ci
attention_type='LAS_LOC_ci'
######'Subsamp_lstm|conv2d||conv1|nothing'
enc_front_end='Subsamp_lstm'
init_full_model='xavier_lin_relu'
tie_dec_weights=1

####tanh|relu
Conv_Act='tanh'
lr_redut_st_th=1
teacher_force=0.6

min_F_bands=5; max_F_bands=30;
time_drop_max=2; time_window_max=1;

weight_noise_flag=1;
reduce_learning_rate_flag=1;
spec_aug_flag=0

# for loading pre-trained weights: set path and if there MUST be the same amount of parameters
pre_trained_weight="0" 
strict_load_weights_flag=0 # keep to zero if you want to train speller on top of no-speller model

plot_fig_validation=0;
plot_fig_training=0;
start_decoding=1
#---------------------------

Word_model_path='/mnt/matylda6/iegorova/speller/Librispeech_960hrs_sentencepiece/Librispeech_960hrs_sentencepiece_word.model'
Char_model_path='/mnt/matylda6/iegorova/speller/Librispeech_960hrs_sentencepiece/Librispeech_960hrs_sentencepiece_char.model'

text_file='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/all_text'

train_path='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/feat_lists/train_960/'
train_clean_360_path='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/feat_lists/train_clean_360/' #_4014/'
dev_clean_path='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/feat_lists/dev_clean/' 
dev_other_path='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/feat_lists/dev_other/'  
test_clean_path='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/feat_lists/test_clean/' 
test_other_path='/mnt/matylda6/iegorova/e2e_hari_v3_speller/data/feat_lists//test_other/' 


model_file="960hrs_10000w_100ch_sh_w_LAS_LOC_ci_2_4_800_lrth30_ls0.1_TF0.6_lr0.0005_clipgrad10_subsamp_maxpool_lstmdr0.3_init_xavier_lin_relu_fbank83_cmvn_warmup20Kexpdecay1000Kmin1e6_decdr0.1_newbob_sp_1oovs_spoovs"
data_dir="/mnt/matylda6/iegorova/speller/models/Librispeech_960hrs_sentencepiece/LIBRISP960hrs_training_data_Nocmvn_10000w_100ch/"



mkdir -pv $data_dir
model_dir="$PPATH/models/$model_file"

weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res

mkdir -pv $model_dir
output_file="$PPATH/log/$model_file".log
log_file="$PPATH/log/$model_file".log
if [[ ! -w $weight_text_file ]]; then touch $weight_text_file; fi
if [[ ! -w $Res_text_file ]]; then touch $Res_text_file; fi

echo "$model_dir"
echo "$weight_file"
echo "$Res_file"


if [ $stage -le 1 ]; then
# #---------------------------------------------------------------------------------------------
##### making the data preperation for the experiment
stdbuf -o0  python /mnt/matylda6/iegorova/speller/scripts/Make_training_scps.py \
                                                --data_dir $data_dir \
                                                --text_file $text_file \
                                                --train_path $train_path \
                                                --dev_path $dev_clean_path \
                                                --Word_model_path $Word_model_path \
                                                --Char_model_path $Char_model_path
echo "$data_dir"
mkdir -pv $data_dir/train_scp_splits/
#rm $data_dir/train_scp_splits/*
split -l 5000 $data_dir/train_scp $data_dir/train_scp_splits/aa_
##---------------------------------------------------------------------------------------------
fi
###scp wrd char

##exit 0

#To avoid stale file handle error
#random_tag="$RANDOM"_temp
#new_data_dir="$data_dir""$random_tag"

#mkdir -pv $new_data_dir
#cp "$data_dir"*_scp "$new_data_dir"
#data_dir="$new_data_dir"/
#echo "$data_dir"
#exit 1

if [ $stage -le 2 ]; then
# #---------------------------------------------------------------------------------------------
##### 
echo "Training started----------:>"
echo $speller_input
stdbuf -o0  python /mnt/matylda6/iegorova/speller/scripts/training.py \
                                                --model_dir $model_dir \
                                                --gpu $gpu \
                                                --max_batch_len $max_batch_len \
                                                --max_batch_label_len $max_batch_label_len \
                                                --max_feat_len $max_feat_len \
                                                --max_label_len $max_label_len \
                                                --tr_disp $tr_disp \
                                                --vl_disp $vl_disp \
                                                --validate_interval $validate_interval\
                                                --max_val_examples $max_val_examples \
                                                --compute_ctc $compute_ctc \
                                                --ctc_weight $ctc_weight \
                                                --learning_rate $learning_rate \
                                                --sp_learning_rate $sp_learning_rate \
                                                --early_stopping $early_stopping \
                                                --clip_grad_norm $clip_grad_norm \
                                                --hidden_size $hidden_size \
                                                --input_size $input_size \
                                                --encoder_layers $encoder_layers \
                                                --lstm_dropout $lstm_dropout \
                                                --kernel_size $kernel_size \
                                                --stride $stride \
                                                --in_channels $in_channels \
                                                --out_channels $out_channels \
                                                --conv_dropout $conv_dropout \
                                                --isresidual $isresidual \
                                                --label_smoothing $label_smoothing \
                                                --attention_type $attention_type \
                                                --enc_front_end $enc_front_end \
                                                --init_full_model $init_full_model \
                                                --Conv_Act $Conv_Act \
                                                --lr_redut_st_th $lr_redut_st_th \
                                                --teacher_force $teacher_force \
                                                --min_F_bands $min_F_bands \
                                                --max_F_bands $max_F_bands \
                                                --time_drop_max $time_drop_max \
                                                --time_window_max $time_window_max \
                                                --weight_noise_flag $weight_noise_flag \
                                                --reduce_learning_rate_flag $reduce_learning_rate_flag \
                                                --spec_aug_flag $spec_aug_flag \
                                                --pre_trained_weight $pre_trained_weight \
                                                --plot_fig_validation $plot_fig_validation \
                                                --plot_fig_training $plot_fig_training \
                                                --data_dir $data_dir \
                                                --Word_model_path $Word_model_path \
                                                --Char_model_path $Char_model_path \
                                                --weight_text_file $weight_text_file \
                                                --Res_text_file $Res_text_file \
                                                --optimizer $optimizer \
                                                --lstm_proj_act $lstm_proj_act \
                                                --pretrain_encoder $pretrain_encoder \
                                                --warmup_steps $warmup_steps \
                                                --sp_warmup_steps $sp_warmup_steps \
                                                --init_lr $init_lr \
                                                --pret_enc_lay_init $pret_enc_lay_init \
                                                --add_layer_after_steps $add_layer_after_steps \
                                                --apply_cmvn $apply_cmvn \
                                                --Dec_model $Dec_model \
                                                --optimizer_style $optimizer_style \
                                                --half_period $half_period \
                                                --min_lr $min_lr \
                                                --start_decay $start_decay \
                                                --dec_dropout $dec_dropout \
                                                --decoder_layers $decoder_layers \
                                                --tie_dec_weights $tie_dec_weights\
                                                --use_speller $use_speller \
                                                --cond_word_on_speller $cond_word_on_speller \
                                                --char_label_smoothing $char_label_smoothing \
                                                --spell_everything $spell_everything \
                                                --speller_cost_weight $speller_cost_weight \
                                                --norm_spell_loss $norm_spell_loss \
                                                --reinitialize_speller_always $reinitialize_speller_always \
                                                --strict_load_weights_flag $strict_load_weights_flag \
                                                --freeze_all_but_speller $freeze_all_but_speller \
                                                --unfreeze_emb $unfreeze_emb \
                                                --num_OOV_embs $num_OOV_embs \
                                                --speller_input $speller_input 
fi
exit 0


# if [ $stage -le 3 ]; 
# then
# gpu=0
# pre_trained_weight="0"
# #model_epoch_38_sample_4999_232.27728486328124___264.8979269988345__0.10355194423918009
# #######this should have at maximum number of files to decode if you want to decode all the file then this should be length of lines in scps
# max_jobs_to_decode=3000 
# mem_req_decoding=10G

# gamma=1.0
# len_pen=0.5

# #dev_clean="/mnt/matylda3/vydana/benchmarking_datasets/Librispeech_V2/scp_files/No_cmvn/dev_clean/"
# #dev_other="/mnt/matylda3/vydana/benchmarking_datasets/Librispeech_V2/scp_files/No_cmvn/dev_other/"
# #test_clean="/mnt/matylda3/vydana/benchmarking_datasets/Librispeech_V2/scp_files/No_cmvn/test_clean/"
# #test_other="/mnt/matylda3/vydana/benchmarking_datasets/Librispeech_V2/scp_files/No_cmvn/test_other/"


# dev_clean="/mnt/scratch02/tmp/vydana/data/Librispeech_data2.0_fbank83_without_CMVN/dev_clean/"
# dev_other="/mnt/scratch02/tmp/vydana/data/Librispeech_data2.0_fbank83_without_CMVN/dev_other/"
# test_clean="/mnt/scratch02/tmp/vydana/data/Librispeech_data2.0_fbank83_without_CMVN/test_clean/"
# test_other="/mnt/scratch02/tmp/vydana/data/Librispeech_data2.0_fbank83_without_CMVN/test_other/"

# for test_fol in $dev_clean $dev_other $test_clean $test_other
# #$test_clean $test_other
# #$dev_clean $dev_other $test_clean $test_other 
# do
# D_path=${test_fol%*/}
# D_path=${D_path##*/}
# echo "$test_fol"
# echo "$D_path"

# #for len_pen in $(seq 0.6 0.1 1)
# #do
# #for gamma in $(seq 0.5 0.5 2)
# #do

# for beam in 10
# do 
# decoding_tag="_decoding_v1_beam_$beam""_$D_path""_gamm$gamma""len_pen$len_pen"
# log_path="$model_dir"/decoding_log_$decoding_tag
# echo "$log_path"

# mkdir -pv "$log_path"
# mkdir -pv "$log_path/scoring"
# mkdir -pv "$model_dir/decoding_files/plots"


# if [ $only_scoring != 'True' ]; 
# then

# for max_jobs in 1 $max_jobs_to_decode
# do
# /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/queue.pl \
#         --max-jobs-run $max_jobs_to_decode \
#         -q all.q@@stable,all.q@@blade \
#         --mem $mem_req_decoding \
#         -l matylda3=0.01,ram_free=$mem_req_decoding,tmp_free=10G \
#         JOB=1:$max_jobs \
#         -l 'h=!blade063' \
#         $log_path/decoding_job.JOB.log \
#         python /mnt/matylda3/vydana/HOW2_EXP/Timit/TIMIT_Att_V1_LSTMSS_decoding_v2.py \
#         --gpu $gpu \
#         --model_dir $model_dir \
#         --Decoding_job_no JOB \
#         --beam $beam \
#         --gamma $gamma \
#         --len_pen $len_pen\
#         --dev_path $test_fol \
#         --weight_text_file $weight_text_file\
#         --Res_text_file $Res_text_file\
#         --text_file $text_file \
#         --pre_trained_weight $pre_trained_weight
# done

# . /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/path.sh
# . /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/SCORING_HELPER.sh $log_path


# else

# . /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/path.sh
# . /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/SCORING_HELPER.sh $log_path

# fi

# done
# done


# fi
