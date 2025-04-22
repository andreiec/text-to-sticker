import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import os
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from utils.dataset import EmojiDataset
from utils.utils import load_checkpoint, save_checkpoint, log_reconstructions


def train_step(batch, encoder, decoder, optimizer, device):
    images = batch['image'].to(device)
    noise = torch.randn(images.size(0), 4, 32, 32).to(device)

    latents = encoder(images, noise)
    recons = decoder(latents)

    loss = F.mse_loss(recons, images)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(encoder, decoder, dataloader, optimizer, device, start_epoch=0, epochs=10):
    encoder.train()
    decoder.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"Epoch {epoch + 1} / {start_epoch + epochs}")
        pbar = tqdm(dataloader, ncols=150)
        total_loss = 0.0
        num_batches = 0

        for batch in pbar:
            loss = train_step(batch, encoder, decoder, optimizer, device)
            total_loss += loss
            num_batches += 1
            pbar.set_postfix({'vae_loss': f"{loss:>8.6f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} done | avg vae loss: {avg_loss:.6f}")

        log_reconstructions(encoder, decoder, dataloader, device, epoch, save=True, save_path='samples/vae/recons')

        if (epoch + 1) % 5 == 0 or (epoch + 1) == (start_epoch + epochs):
            save_checkpoint({"encoder": encoder, "decoder": decoder}, optimizer, epoch, f"checkpoints/vae/vae_epoch_{epoch+1:04d}.pth", vae_only=True)
        
        encoder.train()
        decoder.train()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    dataset = EmojiDataset(project_root / 'data' / 'emoji_dataset_128x128' / 'emoji_dataset.json', image_size=128, tokenize=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    models = {
        'encoder': encoder,
        'decoder': decoder
    }

    start_epoch = 0
    checkpoint_path = project_root / 'checkpoints/vae/vae_epoch_0020.pth'
    resume = True

    if resume:
        start_epoch = load_checkpoint(models, optimizer, checkpoint_path) + 1
        print(f"Resumed from checkpoint: epoch {start_epoch}")

    train(encoder, decoder, dataloader, optimizer, device, start_epoch=start_epoch, epochs=15)
