import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib
matplotlib.use('Agg')

import os
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from utils.dataset import EmojiDataset
from utils.scheluder import KLAnnealingScheduler
from utils.visualisation import sample_from_vae, log_reconstructions_vae
from utils.utils import compute_recon_loss, kl_divergence, load_checkpoint, log_metrics, reparameterize, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train a VAE on emoji dataset')
    parser.add_argument('--model_name', type=str, default='vae-1.0')
    parser.add_argument('--data_json', type=str, default='data/emoji_dataset_128x128/emoji_dataset.json', help='Path to dataset JSON')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--beta_max', type=float, default=1e-4, help='Maximum weight for KL term')
    parser.add_argument('--beta_strategy', type=str, choices=['constant', 'linear', 'sigmoid'], default='sigmoid', help='KL annealing strategy')
    parser.add_argument('--beta_mid', type=int, default=20, help='Midpoint epoch for sigmoid annealing')
    parser.add_argument('--beta_steepness', type=float, default=0.3, help='Steepness for sigmoid annealing')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Optional LR warmup steps for scheduler')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/vae')
    parser.add_argument('--checkpoint_name', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_samples', action='store_true', help='Sample VAE every epoch')
    parser.add_argument('--log_recons', action='store_true', help='Plot reconstructions')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train(
    args,
    encoder,
    decoder,
    dataloader,
    optimizer,
    latent_shape,
    device,
    beta_scheduler,
    lr_scheduler=None,
    start_epoch=0
):
    encoder.train()
    decoder.train()

    scaler = GradScaler()
    total_steps = len(dataloader)
    latent_dims = latent_shape[0] * latent_shape[1] * latent_shape[2]

    for epoch in range(start_epoch, args.epochs):
        beta = args.beta_max
        epoch_loss = epoch_recon = epoch_kl = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=160)

        for step, batch in enumerate(pbar):
            with autocast(device_type=device.type):
                images = batch['image'].to(device)

                mu, logvar = encoder(images)
                z = reparameterize(mu, logvar)
                recons = decoder(z)

                recon_loss = compute_recon_loss(images, recons)
                kl_loss = kl_divergence(mu, logvar)
                loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler:
                lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

            pbar.set_postfix({
                'loss': f"{loss:.6f}",
                'recon': f"{recon_loss:.6f}",
                'kl': f"{kl_loss:.6f}",
                'beta': f"{beta:.1e}",
                'nats/d': f"{kl_loss.item() / latent_dims:.4e}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}" if lr_scheduler else 'NA'
            })

        avg_loss = epoch_loss / total_steps
        avg_recon = epoch_recon / total_steps
        avg_kl = epoch_kl / total_steps

        metrics = {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl,
            'nats/dim': kl_loss.item() / latent_dims,
            'beta': beta,
            'lr': optimizer.param_groups[0]['lr']
        }

        os.makedirs(args.log_dir, exist_ok=True)
        log_metrics(metrics, log_path=os.path.join(args.log_dir, f"{args.model_name}.txt"), epoch=epoch+1)

        recon_dir = f"samples/vae/{args.model_name}/recons"
        sample_dir = f"samples/vae/{args.model_name}/samples"
        ckpt_dir = os.path.join(args.checkpoint_path, args.model_name)

        os.makedirs(ckpt_dir, exist_ok=True)

        if args.log_samples:
            sample_from_vae(encoder, decoder, latent_shape, device, num_samples=8, save=True, save_path=os.path.join(sample_dir, f"epoch_{epoch+1:04d}.png"))

        if args.log_recons:
            log_reconstructions_vae(encoder, decoder, dataloader, device, epoch, save=True, save_path=recon_dir)

        if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
            ckpt_path = os.path.join(ckpt_dir, f"vae_epoch_{epoch+1:04d}.pth")
            save_checkpoint({'encoder': encoder, 'decoder': decoder}, optimizer, scaler, epoch, ckpt_path, vae_only=True)

    print('Training complete.')


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 128, 128, device=device)
        mu_dummy, _ = encoder(dummy)

    latent_c, latent_h, latent_w = mu_dummy.shape[1:]
    latent_shape = (latent_c, latent_h, latent_w)

    project_root = Path(__file__).resolve().parents[1]
    data_json = project_root / args.data_json
    dataset = EmojiDataset(data_json, image_size=args.image_size, tokenize=False, augment=args.augment)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
    )

    lr_scheduler = None

    if args.warmup_steps:
        total_iters = args.epochs * len(dataloader)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)

    beta_scheduler = KLAnnealingScheduler(max_beta=args.beta_max, strategy=args.beta_strategy, mid_epoch=args.beta_mid, steepness=args.beta_steepness)

    start_epoch = 0

    if args.resume:
        ckpt_path = Path(args.checkpoint_path) / args.model_name / args.checkpoint_name

        if ckpt_path.exists():
            start_epoch = load_checkpoint({'encoder':encoder, 'decoder':decoder}, optimizer, ckpt_path) + 1
            print(f"Resumed from epoch {start_epoch}.")

    train(
        args,
        encoder,
        decoder,
        dataloader,
        optimizer,
        latent_shape,
        device,
        beta_scheduler,
        lr_scheduler,
        start_epoch
    )


if __name__ == '__main__':
    main()
