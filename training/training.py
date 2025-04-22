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

    return {'total': loss.item(), 'diff': diffusion_loss.item(), 'recon': recon_loss.item()}


def train(model_dict, dataloader, optimizer, scheduler, device, epochs=10, start_epoch=0, recon_loss_weight=1.0):
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(start_epoch, epochs):

        print(f"Epoch {epoch + 1}")
        pbar = tqdm(dataloader, ncols=120)

        total_loss_sum = 0.0
        diff_loss_sum = 0.0
        recon_loss_sum = 0.0
        num_batches = 0

        for batch in pbar:
            losses = train_step(batch, model_dict, scheduler, optimizer, device, recon_loss_weight)

            total_loss_sum += losses['total']
            diff_loss_sum += losses['diff']
            recon_loss_sum += losses['recon']
            num_batches += 1

            pbar.set_postfix({
                "total": f"{losses['total']:>8.6f}",
                "diff": f"{losses['diff']:>8.6f}",
                "recon": f"{losses['recon']:>8.6f}",
            })
        
        avg_total = total_loss_sum / num_batches
        avg_diff = diff_loss_sum / num_batches
        avg_recon = recon_loss_sum / num_batches

        print(f"Epoch {epoch + 1} done | total: {avg_total:.6f} | diff: {avg_diff:.6f} | recon: {avg_recon:.6f}")


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
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }, f'checkpoints/epoch_{epoch+1:04d}.pth')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    tokenizer = CLIPTokenizer(
        vocab_file=str(project_root / 'data' / 'tokenizer' / 'vocab.json'),
        merges_file=str(project_root / 'data' / 'tokenizer' / 'merges.txt'),
    )

    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)

    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    diffusion = Diffusion().to(device)

    # Load dataset
    dataset = EmojiDataset(project_root / 'data' / 'emoji_dataset_128x128' / 'emoji_dataset.json', image_size=128, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # Load number generator and scheduler
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
        
    scheduler = DDPMSampler(generator)
    scheduler.set_inference_timesteps(num_inference_steps=50)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion.parameters()),
        lr=1e-4
    )

    start_epoch = 0
    checkpoint_path = 'checkpoints/epoch_0005.pth'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        diffusion.load_state_dict(checkpoint['diffusion'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch']
        print(f'Resumed from checkpoint at epoch {start_epoch}')

    models = {
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
        'text_encoder': text_encoder
    }

    train(models, dataloader, optimizer, scheduler, device, epochs=10, start_epoch=5)
