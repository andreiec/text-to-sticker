#!/bin/bash

MODEL_NAME="diffusion-1.7"
VAE_CKPT="checkpoints/vae/vae-b-2.2/vae_epoch_0070.pth"
DIFFUSION_CKPT="checkpoints/diffusion/diffusion-1.7/epoch_0030.pth"
EPOCHS=60
BATCH_SIZE=8
LR=1e-5
LR_SCHEDULER="cosine"
WARMUP_STEPS=300
NUM_TRAIN_STEPS=1000
NUM_INFER_STEPS=30
AUGMENT=false
RECON_LOSS_WEIGHT=0.0
FREEZE_VAE=true
FINETUNE_TEXT=false
LOG_SAMPLES=true
LOG_RECONS=true
RESUME=true
SEED=42


CMD="/opt/conda/envs/py_3.10/bin/python /workspace/training/training.py \
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
