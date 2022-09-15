#!/usr/bin/bash

#. /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/path.sh
. /mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/Basic_Attention_V1/path.sh

scoring_pth='/mnt/matylda6/iegorova/e2e_hari_v3_speller/scripts/Basic_Attention_V1/utils/scoring/' #/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/scoring'

GT=$1
hyp=$2
OUT_PTH=$3

GT_sc=$4
hyp_sc=$5



#=============================================================================================
#-------------------
align-text ark,t:$GT ark,t:$hyp ark,t:$OUT_PTH/alignment.txt_val
cat $OUT_PTH/alignment.txt_val | $scoring_pth/wer_per_utt_details.pl > $OUT_PTH/per_utt_alignment.txt_val
#cat $OUT_PTH/alignment.txt_val | $scoring_pth/wer_report.pl > $OUT_PTH/wer_report.txt_val
cat $OUT_PTH/per_utt_alignment.txt_val | $scoring_pth/wer_ops_details.pl | sort -nrk4 > $OUT_PTH/wer_ops_details.txt_val
compute-wer --text --mode=present ark:$GT  ark,t:$hyp 2>&1> $OUT_PTH/wer_val
#-------------------


# cat $GT | awk '{print $2 "("$1")"}' > $OUT_PTH/GT_temp
# cat $hyp | awk '{print $2 "("$1")"}' > $OUT_PTH/hyp_temp

#=============================================================================================
scoring_pth='/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/scoring/'
#-------------------
#/mnt/matylda3/vydana/eesen/tools/sctk/bin/sclite -r $GT_sc -h $hyp_sc -i rm -o all stdout > $OUT_PTH/result.wrd.txt
#/mnt/matylda3/vydana/eesen/tools/sctk/bin/sclite -r $GT_sc -h $hyp_sc -i rm -o dtl stdout > $OUT_PTH/result.wrd.dtl
#/mnt/matylda3/vydana/eesen/tools/sctk/bin/sclite -r $GT_sc -h $hyp_sc -i rm -o pralign stdout > $OUT_PTH/result.wrd.pralign
#--------------------
#dtl, lur, pralign, prf, rsum, sgml, spk, snt, sum, wws 
