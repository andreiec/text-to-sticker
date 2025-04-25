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
from utils.scheluder import KLAnnealingScheduler
from utils.visualisation import sample_from_vae, log_reconstructions_vae
from utils.utils import kl_divergence, load_checkpoint, log_metrics, reparameterize, save_checkpoint


def train_step(batch, encoder, decoder, optimizer, device, beta=1e-4):
    images = batch['image'].to(device)

    mu, logvar = encoder(images)
    z = reparameterize(mu, logvar)
    recons = decoder(z)

    recon_loss = F.mse_loss(recons, images)
    kl_loss = kl_divergence(mu, logvar)

    loss = recon_loss + beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), recon_loss.item(), kl_loss.item()


def train(model_name, encoder, decoder, dataloader, optimizer, device, start_epoch=0, epochs=10, beta=1e-4, beta_scheduler=None, lr_scheduler=None):
    encoder.train()
    decoder.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"Epoch {epoch + 1} / {start_epoch + epochs}")

        pbar = tqdm(dataloader, ncols=160)
        beta = beta_scheduler(epoch) if beta_scheduler else beta
        total_loss = total_recon = total_kl = 0.0

        for batch in pbar:
            loss, recon, kl = train_step(batch, encoder, decoder, optimizer, device, beta)

            total_loss += loss
            total_recon += recon
            total_kl += kl

            if lr_scheduler:
                lr_scheduler.step()

            pbar.set_postfix({
                'loss': f"{loss:>8.6f}",
                'recon': f"{recon:>8.6f}",
                'kl': f"{kl:>8.6f}",
                'kl_scaled': f"{kl*beta:>8.6f}",
                'lr': f"{lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr']:>8.6f}",
                'beta': f"{beta:>8.6f}"
            })

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        print(f"Epoch {epoch + 1} done | loss: {avg_loss:.6f} | recon: {avg_recon:.6f} | kl: {avg_kl:.6f}")

        reconstructions_path = f"samples/vae-b/{model_name}/recons"
        sample_path = f"samples/vae-b/{model_name}/samples/epoch_{epoch + 1:04d}.png"
        checkpoint_path = f"checkpoints/vae-b/{model_name}/vae_epoch_{epoch+1:04d}.pth"
        log_path = f"logs/{model_name}.txt"

        metrics = {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl,
            'kl_scaled': avg_kl * beta,
            'lr': lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr'],
            'beta': beta
        }

        log_metrics(metrics, log_path=log_path, epoch=epoch + 1)
        log_reconstructions_vae(encoder, decoder, dataloader, device, epoch, save=True, save_path=reconstructions_path)
        sample_from_vae(decoder, device, num_samples=8, save=True, save_path=sample_path)

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

    models = {
        'encoder': encoder,
        'decoder': decoder
    }

    model_name = 'b-sigmoid-3'
    checkpoint_path = project_root / 'checkpoints/vae-b' / model_name / 'vae_epoch_0020.pth'

    start_epoch = 0
    total_epochs = 70
    resume = False

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_epochs * len(dataloader))
    beta_scheduler = KLAnnealingScheduler(max_beta=1e-4, strategy='sigmoid', mid_epoch=20, steepness=0.3)

    if resume:
        start_epoch = load_checkpoint(models, optimizer, checkpoint_path) + 1
        print(f"Resumed from checkpoint: epoch {start_epoch}")

    train(model_name,
          encoder,
          decoder,
          dataloader,
          optimizer,
          device,
          start_epoch=start_epoch,
          epochs=total_epochs,
          lr_scheduler=None,
          beta_scheduler=beta_scheduler
    )
