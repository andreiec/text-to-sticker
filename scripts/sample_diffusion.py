import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import os
import argparse
import torch

from torchvision.utils import make_grid, save_image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import EulerDiscreteScheduler
from typing import Any

from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from src.diffusion import Diffusion
from utils.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from trained emoji diffusion model')
    parser.add_argument('--ckpt', type=str, default='', help='path to diffusion checkpoint')
    parser.add_argument('--prompts', nargs='+', default=[''], help='text prompts')
    parser.add_argument('--gscale', type=float, default=5.0, help='guidance scale')
    parser.add_argument('--steps', type=int, default=50, help='number of inference steps')
    parser.add_argument('--out', type=str, default='samples/diffusion_sample.png', help='where to save the output grid')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    return parser.parse_args()


def sample_diffusion(
    models: dict,
    scheduler: Any,
    tokenizer: CLIPTokenizer,
    prompts: list[str],
    guidance_scale: float,
    output_path: str,
    device: torch.device
):

    encoder = models['encoder']
    decoder = models['decoder']
    diffusion = models['diffusion']
    text_encoder = models['text_encoder']

    encoder.eval()
    decoder.eval()
    diffusion.eval()
    text_encoder.eval()

    batch_size = len(prompts)
    token_ids = tokenizer(prompts, return_tensors='pt', padding='max_length', max_length=77, truncation=True).input_ids.to(device)

    with torch.no_grad():
        context = text_encoder(token_ids).last_hidden_state
        uncond_tokens = tokenizer([''] * batch_size, padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to(device)
        uncond_context = text_encoder(uncond_tokens).last_hidden_state

        latents = torch.randn(batch_size, 4, 32, 32, device=device)

        for idx, t in enumerate(scheduler.timesteps):
            t_int = int(t)
            t_tensor = torch.full((batch_size,), t_int, device=device, dtype=torch.long)

            model_input = scheduler.scale_model_input(latents, t)

            eps_uncond = diffusion(model_input, uncond_context, t_tensor)
            eps_cond   = diffusion(model_input, context, t_tensor)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            out = scheduler.step(model_output=eps_pred, timestep=t, sample=latents)
            latents = out.prev_sample
        
        images = decoder(latents)

    images = images.clamp(-1.0, 1.0).add(1.0).div(2.0)
    grid = make_grid(images, nrow=min(batch_size, 4))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(grid, output_path)
    print(f"Sampled {len(images)} images.")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    diffusion = Diffusion().to(device)

    tokenizer = CLIPTokenizer(
        vocab_file=str(project_root / 'data' / 'tokenizer' / 'vocab.json'),
        merges_file=str(project_root / 'data' / 'tokenizer' / 'merges.txt'),
    )

    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)

    models = {
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
        'text_encoder': text_encoder,
    }

    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule='scaled_linear', num_train_timesteps=1000)
    scheduler.set_timesteps(args.steps, device=device)

    load_checkpoint(models, optimizer=None, path=args.ckpt)
    sample_diffusion(models, scheduler, tokenizer, args.prompts, args.gscale, args.out, args.device)


if __name__ == '__main__':
    main()
