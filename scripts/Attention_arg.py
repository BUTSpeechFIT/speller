#!/usr/bin/pyhton
import sys
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("--gpu",metavar='',type=int,default='0',help="use gpu flag 0|1")
#---------------------------

###model_parameter
parser.add_argument("--hidden_size",metavar='',type=int,default='320',help="Space token for the Char model")
parser.add_argument("--emb_dim",metavar='',type=int,default='800',help="Dimension of embeddings")
parser.add_argument("--input_size",metavar='',type=int,default='249',help="Space token for the Char model")
parser.add_argument("--lstm_dropout",metavar='',type=float,default='0.3',help="lstm_dropout")
parser.add_argument("--kernel_size",metavar='',type=int,default='3',help="kernel_size")
parser.add_argument("--stride",metavar='',type=int,default='2',help="stride")
parser.add_argument("--in_channels",metavar='',type=int,default='1',help="in_channels")
parser.add_argument("--out_channels",metavar='',type=int,default='256',help="out_channels")
parser.add_argument("--conv_dropout",metavar='',type=float,default='0.1',help="conv_dropout")
parser.add_argument("--isresidual",metavar='',type=int,default=0,help="isresidual --> 1|0 ")
parser.add_argument("--enc_front_end",metavar='',type=str,default='Subsamp_lstm',help="Subsamp_lstm|conv2d|nothing")
parser.add_argument("--Conv_Act",metavar='',type=str,default='tanh',help="relu|tanh")
parser.add_argument("--lstm_proj_act", metavar='',type=str,default='tanh',help="relu|tanh")
parser.add_argument("--apply_cmvn", metavar='',type=int,default=1,help="1|0")
parser.add_argument("--dec_dropout", metavar='',type=float,default=0.1,help="1|0")
parser.add_argument("--tie_dec_weights", metavar='',type=int,default=0,help="1|0")



###optimizer parameters
parser.add_argument("--optimizer_style", metavar='',type=str,default='exp_decay',help="exp_decay")
parser.add_argument("--half_period", metavar='',type=int,default=10000,help="1|0")
parser.add_argument("--new_bob_decay",metavar='',type=int,default='0',help="Value of new_bob_decay ")
parser.add_argument("--warmup_steps",metavar='',type=int,default='10000',help="Value of new_bob_decay ")
parser.add_argument("--sp_warmup_steps",metavar='',type=int,default='1000',help="Value of new_bob_decay speller ")
parser.add_argument("--init_lr",metavar='',type=float,default='5e-5',help="Value of new_bob_decay ")
parser.add_argument("--final_lr",metavar='',type=float,default='1e-6',help="Value of new_bob_decay ")
parser.add_argument("--min_lr", metavar='',type=float,default=0.00001,help="1|0")
parser.add_argument("--start_decay", metavar='',type=float,default=0,help="1|0")
parser.add_argument("--optimizer",metavar='',type=str,default='adam',help="adam|adadelta|Linear_warmup_adam|Trapizoidal_warmup_sgd")
parser.add_argument("--sp_optimizer",metavar='',type=str,default='sp_lin_warmup_adam')
parser.add_argument("--steady_steps",metavar='',type=int,default=3000,help="1 ")
parser.add_argument("--cooldown_steps",metavar='',type=int,default=3000,help="1 ")
parser.add_argument("--Norm_training_loss",metavar='',type=int,default=0,help="1 ")
parser.add_argument("--freeze_pretrained_weights",metavar='',type=int,default=0,help="1 ")
parser.add_argument("--unfreeze_emb",metavar='',type=int,default=0,help="1 ")

#freeze_pretrained_weights
#self.char_teacher_force_rate





####schedule 
parser.add_argument("--nepochs",metavar='',type=int,default='100',help="No of epochs")
parser.add_argument("--encoder_dropout",metavar='',type=float,default='0.3',help="encoder dropout ")
parser.add_argument("--encoder_layers",metavar='',type=int,default='4',help="encoder dropout ")
parser.add_argument("--decoder_layers",metavar='',type=int,default='1',help="encoder dropout ")
parser.add_argument("--teacher_force",metavar='',type=float,default='0.6',help="Value of Teacher Force ")
parser.add_argument("--teacher_force_decay_rate",metavar='',type=int,default='1000',help="Value of Teacher Force decay parameter")
parser.add_argument("--learning_rate",metavar='',type=float,default='0.0003',help="Value of learning_rate ")
parser.add_argument("--sp_learning_rate",metavar='',type=float,default='0.0003',help="Value of speller learning_rate ")
parser.add_argument("--clip_grad_norm",metavar='',type=float,default='5',help="Value of clip_grad_norm ")

#parser.add_argument("--char_teacher_force_rate",metavar='',type=int,default='1000',help="speller char_teacher_force_rate")



parser.add_argument("--pretrain_encoder",metavar='',type=int,default=0,help="0|1 ")
parser.add_argument("--pret_enc_lay_init",metavar='',type=int,default=1,help="1 ")
parser.add_argument("--add_layer_after_steps",metavar='',type=int,default=3000,help="1 ")



#####Loss function parameters
parser.add_argument("--label_smoothing",metavar='',type=float,default='0.1',help="label_smoothing float value 0.1")
parser.add_argument("--use_word",metavar='',type=int,default=1,help="use_word flags True|False")
parser.add_argument("--ctc_target_type",metavar='',type=str,default='word',help="ctc_target_type flags word|char")
parser.add_argument("--spell_loss_perbatch",metavar='',type=int,default=0,help="ctc_target_type flags True|False")
parser.add_argument("--attention_type",metavar='',type=str,default='LAS',help="Attention type: LAS|Collin_monotonc|Location_aware")
parser.add_argument("--ctc_weight",metavar='',type=float,default=0.5,help="ctc weight")
parser.add_argument("--compute_ctc",metavar='',type=int,default=1,help="compute ctc flags True|False")
parser.add_argument("--init_full_model",metavar='',type=str,default='xavier_lin_relu',help="xavier_lin_relu|None")


####Training schedule parameters 
parser.add_argument("--no_of_checkpoints",metavar='',type=int,default='2',help="Flag of no_of_checkpoints ")
parser.add_argument("--tr_disp",metavar='',type=int,default='1000',help="Value of tr_disp ")
parser.add_argument("--vl_disp",metavar='',type=int,default='100',help="Value of vl_disp ")
parser.add_argument("--noise_inj_ratio",metavar='',type=float,default='0.1',help="Value of noise_inj_ratio ")
parser.add_argument("--weight_noise_flag",metavar='',type=int,default=0,help="T|F Flag for weight noise injection")

parser.add_argument("--early_stopping",metavar='',type=int,default=1,help="Value of early_stopping ")
parser.add_argument("--early_stopping_checkpoints",metavar='',type=int,default=5,help="Value of early_stopping_checkpoints ")
parser.add_argument("--early_stopping_patience",metavar='',type=int,default=5,help="Value of early_stopping_patience ")

#######
parser.add_argument("--adding_layer_iter",metavar='',type=int,default=10000,help="Value of adding_layer_iter ")
parser.add_argument("--stop_LS_iter",metavar='',type=int,default=50000,help="Value of stop_LS_iter ")
parser.add_argument("--start_TF_iter",metavar='',type=int,default=25000,help="Value of start_TF_iter ")
parser.add_argument("--Dec_model",metavar='',type=str,default='Nothing',help="Value of Dec_model ")
parser.add_argument("--train_full_utt",metavar='',type=int,default=2,help="Value of train_full_utt ")
parser.add_argument("--max_feat_len_pretrain",metavar='',type=int,default=800,help="Value of max_feat_len_pretrain ")
parser.add_argument("--max_label_len_pretrain",metavar='',type=int,default=100,help="Value of max_label_len_pretrain ")
######
######


#---------------------------
parser.add_argument("--reduce_learning_rate_flag",metavar='',type=int,default=1,help="reduce_learning_rate_flag True|False")
parser.add_argument("--lr_redut_st_th",metavar='',type=int,default=3,help="Value of lr_redut_st_th after this epochs the ls reduction gets applied")
parser.add_argument("--CER_hist",metavar='',type=int,default=0,help="if LR decisions are made on CER history instead of WER history")

#---------------------------
parser.add_argument("--use_speller",metavar='',type=int,default=0,help="use_speller")
parser.add_argument("--cond_word_on_speller",metavar='',type=int,default=0,help="")
parser.add_argument("--char_label_smoothing",metavar='',type=int,default=0,help="")
parser.add_argument("--spell_everything",metavar='',type=int,default=1,help="")
parser.add_argument("--speller_cost_weight",metavar='',type=float,default=0,help="")
parser.add_argument("--norm_spell_loss",metavar='',type=int,default=1,help="")
parser.add_argument("--reinitialize_speller_always",metavar='',type=int,default=1,help="")
parser.add_argument("--use_si_ci_spelling",metavar='',type=int,default=0,help="")
parser.add_argument("--word_prior_nomalize_speller",metavar='',type=int,default=0,help="")
parser.add_argument("--char_teacher_force_rate",metavar='',type=float,default=0.6,help="speller char_teacher_force_rate")
parser.add_argument("--only_speller",metavar='',type=int,default=0,help="only_speller")
parser.add_argument("--speller_train_flag",metavar='',type=float,default=0,help="")
parser.add_argument("--freeze_all_but_speller",metavar='',type=float,default=0,help="")
parser.add_argument("--tf_speller",metavar='',type=float,default=0,help="")
parser.add_argument("--speller_input",metavar='',type=str,default="",help="")
parser.add_argument("--num_OOV_embs",metavar='',type=int,default=1,help="")
parser.add_argument("--print_OOV_vec",metavar='',type=int,default=0,help="")
parser.add_argument("--two_spellers",metavar='',type=int,default=0,help="")
parser.add_argument("--prob_oov_emb",metavar='',type=int,default=0,help="")
#----------------------------
####bactching parameers
parser.add_argument("--model_dir",metavar='',type=str,default='models/Default_folder',help="model_dir")
parser.add_argument("--batch_size",metavar='',type=int,default='10',help="batch_size")
parser.add_argument("--max_batch_label_len",metavar='',type=int,default='50000',help="max_batch_label_len")
parser.add_argument("--max_batch_len",metavar='',type=int,default='20',help="max_batch_len")
parser.add_argument("--val_batch_size",metavar='',type=int,default='10',help="val_batch_size")

parser.add_argument("--validate_interval",metavar='',type=int,default='5000',help="steps")
parser.add_argument("--max_train_examples",metavar='',type=int,default='23380',help="steps")
parser.add_argument("--max_val_examples",metavar='',type=int,default='2039',help="steps")

parser.add_argument("--max_feat_len",metavar='',type=int,default='2000',help="max_seq_len the dataloader does not read the sequences longer that the max_feat_len, for memory and some times to remove very long sent for LSTM")
parser.add_argument("--max_label_len",metavar='',type=int,default='200',help="max_labes_len the dataloader does not read the sequences longer that the max_label_len, for memory and some times to remove very long sent for LSTM")

###plot the figures
parser.add_argument("--plot_fig_validation",metavar='',type=int,default=0,help="True|False")
parser.add_argument("--plot_fig_training",metavar='',type=int,default=0,help="True|False")
parser.add_argument("--write_att",metavar='',type=int,default=0,help="True|False")

#**********************************
#Spec Aug
parser.add_argument("--spec_aug_flag",metavar='',type=int,default=0,help="spec_aug_flag")
parser.add_argument("--min_F_bands",metavar='',type=int,default='30',help="min_F_bands")
parser.add_argument("--max_F_bands",metavar='',type=int,default='80',help="max_F_bands")
parser.add_argument("--time_drop_max",metavar='',type=int,default='4',help="time_drop_max")
parser.add_argument("--time_window_max",metavar='',type=int,default='4',help="time_window_max")
#**********************************

#---------------------------
####paths and tokenizers

parser.add_argument("--text_file",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/normalized_text_full_train_text',help="text transcription with dev and eval sentences")
parser.add_argument("--train_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/train/',help="model_dir")
parser.add_argument("--dev_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/dev/',help="model_dir")
parser.add_argument("--test_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/dev/',help="model_dir")
parser.add_argument("--data_dir",metavar='',type=str,default='default_data_dir',help="model_dir")
#---------------------------
parser.add_argument("--Word_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__word.model',help="model_dir")
parser.add_argument("--Char_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__char.model',help="model_dir")

####pretrained weights
#---------------------------
parser.add_argument("--pre_trained_weight",metavar='',type=str,default='0',help="pre_trained_weight if you dont have just give zero ")
parser.add_argument("--retrain_the_last_layer",metavar='',type=str,default='False',help="retrain_final_layer if you dont have just give zero ")
parser.add_argument("--strict_load_weights_flag",metavar='',type=int,default=1,help="retrain_final_layer if you dont have just give zero ")

parser.add_argument("--mode_of_loading_weight",metavar='',type=str,default="Nothing",help="mode_of_loading_weight ")


####load the weights
#---------------------------
parser.add_argument("--weight_text_file",metavar='',type=str,default='weight_folder/weight_file',help="weight_file")
parser.add_argument("--Res_text_file",metavar='',type=str,default='weight_folder/weight_file_res',help="Res_file")


####decoding_parameters
parser.add_argument("--RNNLM_model",metavar='',type=str,default='/mnt/matylda3/vydana/HOW2_EXP/Timit/models/TIMIT_fullnewsetup_2_4dr0.3_LAS_loc_arg_format_V2/model_architecture_',help="")
parser.add_argument("--LM_model",metavar='',type=str,default='None',help="LM_model")
parser.add_argument("--Am_weight",metavar='',type=float,default=1,help="lm_weight a float calue between 0 to 1 --->(Am_weight* Am_pred + (1-Am_weight)*lm_pred)")
parser.add_argument("--beam",metavar='',type=int,default=10,help="beam for decoding")
parser.add_argument("--gamma",metavar='',type=float,default=1,help="gamma (0-2), noisy eos rejection scaling factor while decoding")
parser.add_argument("--len_pen",metavar='',type=float,default=1,help="len_pen(0.5-2), len_pen maximum number of decoding steps")
parser.add_argument("--Decoding_job_no",metavar='',type=int,default=0,help="Res_file")
parser.add_argument("--scp_for_decoding",metavar='',type=int,default=0,help="scp file for decoding")
parser.add_argument("--plot_decoding_pics",metavar='',type=int,default=0,help="T|F")
parser.add_argument("--decoder_plot_name",metavar='',type=str,default='default_folder',help="T|F")
parser.add_argument("--ins_pen",metavar='',type=int,default=0,help="ins_pen")
parser.add_argument("--lm_coupling",metavar='',type=int,default=0,help="lm_coupling")

#---------------------------
parser.add_argument("-v","--verbosity",action="count",help="increase output verbosity")





