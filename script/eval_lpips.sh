#!/bin/bash

### Calculating Variety LPIPS (diversity) ###

### START USAGE ###
# sh script/eval_lpips.sh ${EXPID} ${EPOCH}
### END USAGE ###

EXPID=$1
EPOCH=$2
CUDA_VISIBLE_DEVICES=3 python infer_terse.py --data_root /data/CodeGoat24/new_OPA/ --expid ${EXPID} --epoch ${EPOCH} --eval_type evaluni --repeat 10
CUDA_VISIBLE_DEVICES=3 python eval/lpips_1dir.py -d /data/CodeGoat24/placement_result/${EXPID}/evaluni/${EPOCH}/images/ --expid ${EXPID} --epoch ${EPOCH} --eval_type "evaluni" --repeat 10 --use_gpu
