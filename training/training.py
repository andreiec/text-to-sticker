import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib
matplotlib.use("Agg")

import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

from src.ddpm import DDPMSampler
from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from src.diffusion import Diffusion
from utils.dataset import EmojiDataset
from utils.utils import log_vae_reconstructions, sample_and_log


def train_step(batch, models, scheduler, optimizer, device, recon_loss_weight=1.0):
    encoder = models['encoder']
    decoder = models['decoder']
    diffusion = models['diffusion']
    text_encoder =  models['text_encoder']

    images = batch['image'].to(device)
    tokens = batch['tokens'].to(device)
    context = text_encoder(tokens).last_hidden_state


    noise = torch.randn(images.size(0), 4, 32, 32).to(device)
    latents = encoder(images, noise)

    bsz = latents.size(0)
    true_noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.num_training_steps, (bsz,), device=device)
    noisy_latents = scheduler.add_noise(latents, timesteps, noise=true_noise)

    pred_noise = diffusion(noisy_latents, context, timesteps)
    diffusion_loss = F.mse_loss(pred_noise, true_noise)

    recon_images = decoder(latents)
    recon_loss = F.mse_loss(recon_images, images)

    loss = diffusion_loss + recon_loss_weight * recon_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), diffusion_loss.item(), recon_loss.item()


def train(model_dict, dataloader, optimizer, scheduler, device, epochs=10):
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        pbar = tqdm(dataloader)

        for batch in pbar:
            total_loss, diff_loss, rec_loss = train_step(batch, model_dict, scheduler, optimizer, device)

            pbar.set_postfix({
                "total": f"{total_loss:.4f}",
                "diff": f"{diff_loss:.4f}",
                "recon": f"{rec_loss:.4f}",
            })
        
        sample_and_log(
            diffusion=model_dict['diffusion'],
            decoder=model_dict['decoder'],
            tokenizer=tokenizer,
            text_encoder=model_dict['text_encoder'],
            scheduler=scheduler,
            device=device,
            epoch=epoch
        )

        log_vae_reconstructions(
            encoder=model_dict['encoder'],
            decoder=model_dict['decoder'],
            dataloader=dataloader,
            device=device,
            epoch=epoch
        )

        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'diffusion': diffusion.state_dict(),
            'text_encoder': text_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }, f'checkpoints/epoch_{epoch+1:04d}.pt')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_path = project_root / 'data' / 'tokenizer'

tokenizer = CLIPTokenizer(
    vocab_file=str(tokenizer_path / 'vocab.json'),
    merges_file=str(tokenizer_path / 'merges.txt'),
)

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
encoder = VAE_Encoder().to(device)
decoder = VAE_Decoder().to(device)
diffusion = Diffusion().to(device)

dataset = EmojiDataset(str(project_root / 'data/emoji_dataset_128x128/emoji_dataset.json'), image_size=128, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

generator = torch.Generator(device=device)

seed = 42

if seed is None:
    generator.seed()
else:
    generator.manual_seed(seed)
    
scheduler = DDPMSampler(generator)
scheduler.set_inference_timesteps(num_inference_steps=50)

#optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-4)

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion.parameters()),
    lr=1e-4
)

models = {
    'encoder': encoder,
    'decoder': decoder,
    'diffusion': diffusion,
    'text_encoder': text_encoder
}

train(models, dataloader, optimizer, scheduler, device, epochs=5)
