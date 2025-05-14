import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')

from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from typing import Any

from src.ddpm import DDPMSampler
from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from src.diffusion import Diffusion
from utils.dataset import EmojiDataset
from utils.visualisation import log_reconstructions, sample_and_log
from utils.utils import create_scheduler, load_checkpoint, log_metrics, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for emoji diffusion")
    parser.add_argument("--model_name", type=str, default="diffusion-1.0")
    parser.add_argument("--vae_ckpt", type=str, default="checkpoints/vae-b/b-sigmoid-3/vae_epoch_0070.pth")
    parser.add_argument("--diffusion_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["cosine", "constant", "linear"], help="Type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--num_infer_steps", type=int, default=50)
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument("--recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--freeze_vae", action="store_true")
    parser.add_argument("--finetune_text", action="store_true")
    parser.add_argument('--log_samples', action='store_true', help='Sample diffusion every 5 epochs')
    parser.add_argument('--log_recons', action='store_true', help='Plot reconstructions')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(
    model_name: str,
    models: dict,
    dataloader: DataLoader,
    tokenizer: CLIPTokenizer,
    optimizer: torch.optim.Optimizer,
    train_scheduler: Any,
    infer_scheduler: Any,
    device: torch.device,
    lr_scheduler: Any,
    scaler: GradScaler,
    args,
    start_epoch: int = 0,
):

    encoder = models['encoder']
    decoder = models['decoder']
    diffusion = models['diffusion']
    text_encoder = models['text_encoder']
    total_steps = len(dataloader)

    for epoch in range(start_epoch, args.epochs):
        diffusion.train()

        if args.freeze_vae:
            encoder.eval()
            decoder.eval()
        else:
            encoder.train()
            decoder.train()

        if args.finetune_text:
            text_encoder.train()
        else:
            text_encoder.eval()

        pbar = tqdm(dataloader, ncols=150, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = total_diffusion = total_recon = 0

        for batch in pbar:
            images = batch['image'].to(device)
            tokens = batch['tokens'].to(device)

            with torch.no_grad():
                context = text_encoder(tokens).last_hidden_state

            with autocast(device_type=device.type):
                mu, _ = encoder(images)
                latents = mu * 0.24652 # Funky number hack

                bsz = latents.size(0)
                timesteps = train_scheduler.sample_train_timesteps(bsz, device)
                true_noise = torch.randn_like(latents)
                noisy_latents = train_scheduler.add_noise(latents, timesteps, noise=true_noise)

                pred_noise = diffusion(noisy_latents, context, timesteps)
                diffusion_loss = F.mse_loss(pred_noise, true_noise)

                with torch.no_grad():
                    recon_images = decoder(mu)
                    recon_images = recon_images.clamp(-1, 1)

                recon_loss = F.mse_loss(recon_images, images)
                loss = diffusion_loss # + args.recon_loss_weight * recon_loss

            total_loss += loss
            total_diffusion += diffusion_loss
            total_recon += recon_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            pbar.set_postfix({
                'loss': f"{loss.item():>8.6f}",
                'diff': f"{diffusion_loss.item():>8.6f}",
                'recon': f"{recon_loss.item():>8.6f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_loss = total_loss / total_steps
        avg_diffusion = total_diffusion / total_steps
        avg_recon = total_recon / total_steps

        metrics = {
            'loss': avg_loss.item(),
            'diffusion': avg_diffusion.item(),
            'recon': avg_recon.item(),
            'lr': lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr'],
        }

        log_metrics(metrics, log_path=f"logs/diffusion/{model_name}.txt", epoch=epoch+1)

        if (epoch + 1) % 20 == 0 or (epoch + 1 == args.epochs):
            if args.log_samples:
                sample_and_log(
                    diffusion=diffusion,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    scheduler=infer_scheduler,
                    device=device,
                    epoch=epoch,
                    save=True,
                    save_path=f"samples/diffusion/{model_name}/samples",
                    seed=args.seed
                )

            if args.log_recons:
                log_reconstructions(
                    encoder=encoder,
                    decoder=decoder,
                    dataloader=dataloader,
                    device=device,
                    epoch=epoch,
                    save=True,
                    save_path=f"samples/diffusion/{model_name}/vae_recon"
                )

            ckpt_path = f"checkpoints/diffusion/{model_name}/epoch_{epoch+1:04d}.pth"
            save_checkpoint(models, optimizer, scaler, epoch, ckpt_path)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer(
        vocab_file=str(project_root / 'data' / 'tokenizer' / 'vocab.json'),
        merges_file=str(project_root / 'data' / 'tokenizer' / 'merges.txt'),
    )

    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    diffusion = Diffusion().to(device)

    if args.vae_ckpt and os.path.exists(args.vae_ckpt):
        print('Loaded VAE weights.')
        ckpt = torch.load(args.vae_ckpt, map_location='cpu', weights_only=False)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
  
    if args.freeze_vae:
        for p in encoder.parameters(): p.requires_grad = False
        for p in decoder.parameters(): p.requires_grad = False
    if not args.finetune_text:
        for p in text_encoder.parameters(): p.requires_grad = False


    dataset = EmojiDataset(project_root / 'data' / 'emoji_dataset_128x128' / 'emoji_dataset.json', image_size=128, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    train_scheduler = DDPMSampler(generator, num_training_steps=args.num_train_steps).to(device)
    infer_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False, set_alpha_to_one=False)
    infer_scheduler.set_timesteps(args.num_infer_steps, device=device)

    train_params = diffusion.parameters()

    if not args.freeze_vae:
        train_params = list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion.parameters())

    optimizer = torch.optim.AdamW(train_params, lr=args.lr)
    scaler = GradScaler()

    models = {
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
        'text_encoder': text_encoder,
    }

    start_epoch = 0

    if args.resume and args.diffusion_ckpt and os.path.exists(args.diffusion_ckpt):
        ckpt = torch.load(args.diffusion_ckpt, map_location='cpu', weights_only=False)
        start_epoch = load_checkpoint(models, optimizer, scaler, args.diffusion_ckpt) + 1
        print(f"Resumed from epoch {start_epoch}.")

        for pg in optimizer.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg.get('lr', args.lr)


    total_steps = args.epochs * len(dataloader)

    lr_scheduler = create_scheduler(
        optimizer,
        args.lr_scheduler,
        args.warmup_steps,
        total_steps,
        start_epoch=start_epoch * len(dataloader) if args.resume else -1
    )

    train(
        args.model_name,
        models,
        dataloader,
        tokenizer,
        optimizer,
        train_scheduler,
        infer_scheduler,
        device,
        lr_scheduler,
        scaler,
        args,
        start_epoch,
    )


if __name__ == '__main__':
    main()
