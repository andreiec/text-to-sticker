import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 128
HEIGHT = 128
LATENT_WIDTH = WIDTH // 4
LATENT_HEIGHT = HEIGHT // 4


def generate(prompt,
             uncond_prompt,
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name='ddpm',
             num_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):

    with torch.no_grad():
        if idle_device: 
            to_idle: lambda x: x.to(idle_device)
        else: 
            to_idle: lambda x:x
    
        generator = torch.Generator(device=device)
        
        if not seed: generator.seed()
        else: generator.manual_seed(seed)
        
        clip = models['clip']
        clip.to(device)
        
        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            cond_context = clip(cond_tokens)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        
        to_idle(clip)

        if sampler_name != 'ddpm':
            raise ValueError('Uknown sampler.')

        sampler = DDPMSampler(generator)
        sampler.set_inference_steps(num_inference_steps)

        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
        latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models['diffussion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embeddings(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to('cpu', torch.uint8).numpy()


def rescale(x, old_range, new_range, clamp):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x = x - old_min
    x = x * (new_max - new_min) / (old_max - old_min)
    x = x + new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x


def get_time_embeddings(timesteps):
    freqs = torch.pow(10000, -torch.arange(start=0, end=100, dtype=torch.float32) / 160)
    x = torch.tensor([timesteps], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
