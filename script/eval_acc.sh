#!/bin/bash

### Calculating Accuracy (plausibility) ###

### START USAGE ###
# sh script/eval_acc.sh ${EXPID} ${EPOCH} ${BINARY_CLASSIFIER}
### END USAGE ###

EXPID=$1
EPOCH=$2
BINARY_CLASSIFIER=$3

cd faster-rcnn
#python generate_tsv.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval" --cuda
python generate_tsv.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
python convert_data.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
cd ..
python eval/simopa_acc.py --checkpoint ${BINARY_CLASSIFIER} --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"

### Uncomment the following lines if you would like to delete faster-rcnn intermediate results ###
rm /data/CodeGoat24/placement_result/${EXPID}/eval/${EPOCH}/eval_roiinfos.csv
rm /data/CodeGoat24/placement_result/${EXPID}/eval/${EPOCH}/eval_fgfeats.npy
rm /data/CodeGoat24/placement_result/${EXPID}/eval/${EPOCH}/eval_scores.npy
rm /data/CodeGoat24/placement_result/${EXPID}/eval/${EPOCH}/eval_feats.npy
rm /data/CodeGoat24/placement_result/${EXPID}/eval/${EPOCH}/eval_bboxes.npy
