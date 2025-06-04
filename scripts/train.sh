#!/bin/bash

MODEL_NAME="diffusion-1.0"
DATA_JSON="data/sticker_dataset_128x128/dataset.json"
IMAGE_DIR="data/sticker_dataset_128x128/images"
BLACKLIST="data/sticker_dataset_128x128/blacklist.txt"
IMAGE_SIZE=128
BATCH_SIZE=32
EPOCHS=80
LR=1e-5
LR_SCHEDULER="cosine"
WARMUP_STEPS=500
NUM_TRAIN_STEPS=1000
NUM_INFER_STEPS=100
AUGMENT=false
RECON_LOSS_WEIGHT=0.0
FREEZE_VAE=true
FINETUNE_TEXT=false
LOG_SAMPLES=true
LOG_RECONS=true
VAE_CKPT="checkpoints/vae/vae-1.0/vae_epoch_0100.pth"
DIFFUSION_CKPT="checkpoints/diffusion/diffusion-1.7/epoch_0030.pth"
RESUME=false
SEED=42


CMD="/venv/main/bin/python /workspace/text-to-emoji/training/training.py \
  --model_name ${MODEL_NAME} \
  --vae_ckpt ${VAE_CKPT} \
  --data_json ${DATA_JSON} \
  --image_dir ${IMAGE_DIR} \
  --blacklist ${BLACKLIST} \
  --image_size ${IMAGE_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --lr_scheduler ${LR_SCHEDULER} \
  --warmup_steps ${WARMUP_STEPS} \
  --num_train_steps ${NUM_TRAIN_STEPS} \
  --num_infer_steps ${NUM_INFER_STEPS} \
  --recon_loss_weight ${RECON_LOSS_WEIGHT} \
  --seed ${SEED}"


if [ "$DIFFUSION_CKPT" != "" ]; then
  CMD="$CMD --diffusion_ckpt ${DIFFUSION_CKPT}"
fi
if [ "$AUGMENT" = true ]; then
  CMD="$CMD --augment"
fi
if [ "$FREEZE_VAE" = true ]; then
  CMD="$CMD --freeze_vae"
fi
if [ "$FINETUNE_TEXT" = true ]; then
  CMD="$CMD --finetune_text"
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
