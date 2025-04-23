import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from utils.dataset import EmojiDataset
from utils.utils import get_kl_beta_linear, get_kl_beta_sigmoid, kl_divergence, load_checkpoint, reparameterize, sample_from_vae_b, save_checkpoint, log_reconstructions_vae


def train_step(batch, encoder, decoder, optimizer, device, beta=1e-4):
    images = batch['image'].to(device)

    mu, logvar = encoder(images)
    z = reparameterize(mu, logvar)
    recons = decoder(z * 0.18215)

    recon_loss = F.mse_loss(recons, images)
    kl_loss = kl_divergence(mu, logvar)

    loss = recon_loss + beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), recon_loss.item(), kl_loss.item()


def train(encoder, decoder, dataloader, optimizer, device, start_epoch=0, epochs=10, beta_schedule='linear'):
    encoder.train()
    decoder.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        if beta_schedule == 'linear':
            beta = get_kl_beta_linear(epoch, warmup_epochs=10, max_beta=1e-2)
        elif beta_schedule == 'sigmoid':
            beta = get_kl_beta_sigmoid(epoch, max_beta=0.05, steepness=0.25, mid_epoch=20)
        else:
            raise ValueError("Unknown beta_schedule: choose 'linear' or 'sigmoid'")

        print(f"Epoch {epoch + 1} / {start_epoch + epochs}")
        pbar = tqdm(dataloader, ncols=160)

        total_loss = total_recon = total_kl = 0.0

        for batch in pbar:
            loss, recon, kl = train_step(batch, encoder, decoder, optimizer, device, beta)

            total_loss += loss
            total_recon += recon
            total_kl += kl

            pbar.set_postfix({
                'loss': f"{loss:>8.6f}",
                'recon': f"{recon:>8.6f}",
                'kl': f"{kl:>8.6f}",
                'kl_scaled': f"{kl*beta:>8.4f}",
                'beta': f"{beta:>8.4f}"
            })

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        print(f"Epoch {epoch + 1} done | loss: {avg_loss:.6f} | recon: {avg_recon:.6f} | kl: {avg_kl:.6f}")

        reconstructions_path = 'samples/vae-b/b-linear-scheduled/recons'
        sample_path = f"samples/vae-b/b-linear-scheduled/samples/epoch_{epoch + 1:04d}.png"
        checkpoint_path = f"checkpoints/vae-b/b-linear-scheduled/vae_epoch_{epoch+1:04d}.pth"

        log_reconstructions_vae(encoder, decoder, dataloader, device, epoch, save=True, save_path=reconstructions_path)
        sample_from_vae_b(decoder, device, num_samples=8, save=True, save_path=sample_path)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == (start_epoch + epochs):
            save_checkpoint({'encoder': encoder, 'decoder': decoder}, optimizer, epoch, checkpoint_path, vae_only=True)
        
        encoder.train()
        decoder.train()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    dataset = EmojiDataset(project_root / 'data' / 'emoji_dataset_128x128' / 'emoji_dataset.json', image_size=128, tokenize=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    models = {
        'encoder': encoder,
        'decoder': decoder
    }

    start_epoch = 0
    checkpoint_path = project_root / 'checkpoints/vae-b/b-linear-scheduled/vae_epoch_0020.pth'
    resume = True

    if resume:
        start_epoch = load_checkpoint(models, optimizer, checkpoint_path) + 1
        print(f"Resumed from checkpoint: epoch {start_epoch}")

    train(encoder, decoder, dataloader, optimizer, device, start_epoch=start_epoch, epochs=40, beta_schedule='sigmoid')
