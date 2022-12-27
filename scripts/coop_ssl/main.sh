#!/bin/bash

# custom config
DATA=/scratch/michal5/coop_data
TRAINER=CoOpSSL
OUTPUT_PREFIX=/home/michal5/coop_output
DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6 
EXTRA_OUTPUT=$7 # class-specific context (False or True)

for SEED in 1 2 3
do
    DIR=${OUTPUT_PREFIX}/${EXTRA_OUTPUT}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}

done