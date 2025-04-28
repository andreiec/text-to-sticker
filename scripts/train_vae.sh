#!/bin/bash

MODEL_NAME="vae-b-2.2"
DATA_JSON="data/emoji_dataset_128x128/emoji_dataset.json"
IMAGE_SIZE=128
BATCH_SIZE=32
EPOCHS=70
LR=1e-4
AUGMENT=false
BETA_MAX=1e-5
BETA_STRATEGY="sigmoid" # or linear
BETA_MID=20
BETA_STEEPNESS=0.25
WARMUP_STEPS=100
CHECKPOINT_PATH="checkpoints/vae"
CHECKPOINT_NAME=""
LOG_DIR="logs"
LOG_SAMPLES=true
LOG_RECONS=true
RESUME=false
SEED=42


CMD="/opt/conda/envs/py_3.10/bin/python /workspace/training/training_vae.py \
  --model_name ${MODEL_NAME} \
  --data_json ${DATA_JSON} \
  --image_size ${IMAGE_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --beta_max ${BETA_MAX} \
  --beta_strategy ${BETA_STRATEGY} \
  --beta_mid ${BETA_MID} \
  --beta_steepness ${BETA_STEEPNESS} \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --log_dir ${LOG_DIR} \
  --seed ${SEED}"


if [ "$WARMUP_STEPS" != "0" ]; then
  CMD="$CMD --warmup_steps ${WARMUP_STEPS}"
fi
if [ "$CHECKPOINT_NAME" != "" ]; then
  CMD="$CMD --checkpoint_name ${CHECKPOINT_NAME}"
fi
if [ "$AUGMENT" = true ]; then
  CMD="$CMD --augment"
fi
if [ "$LOG_SAMPLES" = true ]; then
  CMD="$CMD --log_samples"
fi
if [ "$LOG_RECONS" = true ]; then
  CMD="$CMD --log_recons"
fi
if [ "$RESUME" = true ]; then
  CMD="$CMD --resume"
fi

eval $CMD
