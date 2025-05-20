#!/bin/bash

MODEL_NAME="vae-o1.1"
DATA_JSON="data/sticker_dataset_128x128/dataset.json"
IMAGE_DIR="data/sticker_dataset_128x128/images"
BLACKLIST="data/sticker_dataset_128x128/blacklist.txt"
IMAGE_SIZE=128
BATCH_SIZE=128
EPOCHS=100
LR=1e-4
AUGMENT=false
BETA_MAX=1e-5
BETA_STRATEGY="sigmoid" # or linear
BETA_MID=20
BETA_STEEPNESS=0.25
WARMUP_STEPS=100
CHECKPOINT_PATH="checkpoints/vae"
CHECKPOINT_NAME=""
LOG_DIR="logs/vae"
LOG_SAMPLES=false
LOG_RECONS=true
RESUME=false
SEED=42


CMD="/venv/main/bin/python /workspace/text-to-emoji/training/training_vae.py \
  --model_name ${MODEL_NAME} \
  --data_json ${DATA_JSON} \
  --image_dir ${IMAGE_DIR} \
  --blacklist ${BLACKLIST} \
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
