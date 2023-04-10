#!/bin/bash

### Calculating FID (plausibility) ###

### START USAGE ###
# sh script/eval_fid.sh ${EXPID} ${EPOCH} ${FID_GT_IMGS}
### END USAGE ###
#export TORCH_CUDA_ARCH_LIST="6.0"
EXPID=$1
EPOCH=$2
FID_GT_IMGS=$3

python eval/fid_resize299.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
CUDA_VISIBLE_DEVICES=2 python eval/fid_score.py /data/CodeGoat24/placement_result/${EXPID}/eval/${EPOCH}/images299/ ${FID_GT_IMGS} --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
