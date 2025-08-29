#!/bin/bash


CKPT="checkpoints/diffusion/diffusion-1.0/epoch_0140.pth"
PROMPTS=("happy cat" "a woman with eyeglasses" "sad panda" "dog with blue hat")
GSCALE=3.0
STEPS=40
OUT="samples/diffusion/sample_output.png"
DEVICE="cuda"

/opt/conda/envs/py_3.10/bin/python /workspace/utils/sample_diffusion.py \
  --ckpt "$CKPT" \
  --gscale "$GSCALE" \
  --steps "$STEPS" \
  --out "$OUT" \
  --device "$DEVICE" \
  --prompts "${PROMPTS[@]}"

