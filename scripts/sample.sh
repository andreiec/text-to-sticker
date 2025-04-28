#!/bin/bash


CKPT="checkpoints/diffusion/diffusion-1.7/epoch_0030.pth"
PROMPTS=("happy cat" "robot with wings" "sad panda")
GSCALE=2.5
STEPS=30
OUT="samples/diffusion/sample_output.png"
DEVICE="cuda"

CMD="/opt/conda/envs/py_3.10/bin/python /workspace/scripts/sample_diffusion.py \
  --ckpt ${CKPT} \
  --gscale ${GSCALE} \
  --steps ${STEPS} \
  --out ${OUT} \
  --device ${DEVICE}"

# Add all prompts
for prompt in "${PROMPTS[@]}"
do
  CMD="$CMD --prompts \"$prompt\""
done

eval $CMD
