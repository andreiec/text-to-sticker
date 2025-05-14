#!/bin/bash

MODEL_NAME="diffusion-1.8"
VAE_CKPT="checkpoints/vae/vae-b-3.6/vae_epoch_0100.pth"
DIFFUSION_CKPT="checkpoints/diffusion/diffusion-1.7/epoch_0030.pth"
EPOCHS=200
BATCH_SIZE=16
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
RESUME=false
SEED=42


CMD="/venv/main/bin/python /workspace/text-to-emoji/training/training.py \
  --model_name ${MODEL_NAME} \
  --vae_ckpt ${VAE_CKPT} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
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
