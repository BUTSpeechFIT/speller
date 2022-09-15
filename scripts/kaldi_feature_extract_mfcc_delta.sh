#!/bin/bash
# Copyright 2013     Yajie Miao     Carnegie Mellon University
# Apache 2.0


## Begin configuration section.  
stage=1
every_nth_frame=1 


nj=100
cmd=run.pl
norm_vars=true
do_concat=true 

# Config for splitting pfile into training and valid set; not used for SWBD
do_split=false  # whether to do pfile splitting
pfile_unit_size=40 # the number of utterances of each small unit into which the whole pfile is chopped 
cv_ratio=0.05 # the ratio of CV data
## End configuration options.


echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "usage: steps/build_nnet_pfile.sh <data-dir> data/train_si284_mfcc_delta"
   echo "e.g.:  steps/build_nnet_pfile.sh data/train exp/tri4_ali exp/tri4_pfile"
   echo "main options (for others, see top of script file)"
   echo "  --stage <stage>                                  # starts from which stage"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
#alidir=$2
dir=$2
name=`basename $data`
#nj=`cat $alidir/num_jobs` || exit 1;
sdata=$data/split$nj

export PATH=$PATH:$PPATH


#echo $norm_vars > $dir/norm_vars

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Setup features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"

# Add first and second-order deltas if needed
#feats="$feats add-deltas ark:- ark:-"

## Get the dimension of the features
$cmd JOB=1:1 $dir/log/get_feat_dim.log \
    feat-to-dim "$feats subset-feats --n=1 ark:- ark:- |" ark,t:$dir/feat_dim || exit 1;
feat_dim=`cat $dir/feat_dim | awk '{print $NF}'`
echo "$0: network inputs have the dimension of $feat_dim"



#data/train_si284_mfcc_delta
#
####extra steps added fro feature extraction#######################################
name=`basename $data`
logdir=$dir
feadir=$logdir
$cmd JOB=1:$nj $logdir/make_features_pdnn.JOB.log \
  copy-feats "$feats" \
  ark,scp:$feadir/feats_pdnn_$name.JOB.ark,$feadir/feats_pdnn_$name.JOB.scp || exit 1;
###################################################################################

echo "$0: done creating features."

exit 0;

