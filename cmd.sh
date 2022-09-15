# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# On Eddie use:
#export train_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=08:00:00"
#export decode_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=05:00:00 -pe memory-2G 4"
#export highmem_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=05:00:00 -pe memory-2G 4"
#export scoring_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=00:20:00"

# JSALT2015 workshop, cluster AWS-EC2, (setup from Vijay)
export train_cmd="queue.pl -l --mem 1G"
export decode_cmd="queue.pl -l --mem 2G"
export highmem_cmd="queue.pl -l arch=*64* --mem 4G"
export scoring_cmd="queue.pl -l arch=*64*"
export cuda_cmd="queue.pl --gpu 1 -l mem_free=20G,ram_free=20G"
export cntk_decode_cmd="queue.pl -l arch=*64* --mem 1G -pe smp 2"

# To run locally, use:
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export highmem_cmd=run.pl
#export cuda_cmd=run.pl

if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
  # BUT cluster:
  queue="all.q@@stable" #@stable"
#  queue="all.q@blade0[56789]*,all.q@blade102"
  long_queue="long.q@@stable"
  gpu_queue="long.q@supergpu*,long.q@dellgpu*,long.q@pcspeech-gpu,long.q@pcgpu*"
  storage="matylda6"
  export train_cmd="queue.pl -q $queue -l ram_free=1500M,mem_free=1500M,${storage}=0.1"
  export decode_cmd="queue.pl -q $queue -l ram_free=50G,mem_free=50G,${storage}=0.1"
  export decode_long_cmd="queue.pl -q $long_queue -l ram_free=10G,mem_free=10G,${storage}=1"
  export cuda_cmd="queue.pl -q $gpu_queue --gpu 1 -l mem_free=10G,ram_free=10G,gpu_ram=14G,${storage}=1" #-l gpu=1"
fi 

