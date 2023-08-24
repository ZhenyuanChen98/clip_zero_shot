#!/bin/bash

# custom config
DATA=/data/clip_zero_shot/
TRAINER=Caption_distill_double

DATASET=COCO2014_distill
CFG=rn50_coco  # config file
CTP=end # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
run_ID=coco_chatglm_valid_20w


export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

### 训练 & 测试
for SEED in 1 # 2 3
do
    DIR=output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    echo "Run this job andsave the output to ${DIR}"
    python train_caption.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.Caption.N_CTX ${NCTX} \
    TRAINER.Caption.CSC ${CSC} \
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} # \
done

### 从现有的训练模型运行测试
# for SEED in 1
# do
#     DIR=output/evaluation/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
#     echo "Run this job andsave the output to ${DIR}"
#     python train_caption.py \
#     --root ${DATA} \
#     --seed ${SEED} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#     --output-dir ${DIR}\
#     --model-dir output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}\
#     --eval-only \
#     TRAINER.Caption.N_CTX ${NCTX} \
#     TRAINER.Caption.CSC ${CSC} \
#     TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP}
#     # fi
# done



### chatglm gen
# bash main.sh pazhou_distill_chatglm rn50 end 16 False pazhou_chatglm_valid 0
