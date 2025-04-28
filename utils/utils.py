import os
import math
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup


def save_checkpoint(models, optimizer, scaler, epoch, path, vae_only=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        'encoder': models['encoder'].state_dict(),
        'decoder': models['decoder'].state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }

    if not vae_only:
        state['diffusion'] = models['diffusion'].state_dict()

    torch.save(state, path)


def load_checkpoint(models, optimizer, scaler, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    models['encoder'].load_state_dict(checkpoint['encoder'])
    models['decoder'].load_state_dict(checkpoint['decoder'])

    if 'diffusion' in checkpoint and 'diffusion' in models:
        models['diffusion'].load_state_dict(checkpoint['diffusion'])

    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()


def get_kl_beta_linear(epoch, warmup_epochs=10, max_beta=1e-2):
    return min(max_beta, max_beta * epoch / warmup_epochs)


def get_kl_beta_sigmoid(epoch, max_beta=1e-2, steepness=0.25, mid_epoch=10):
    return max_beta / (1 + math.exp(-steepness * (epoch - mid_epoch)))


def log_metrics(metrics: dict, log_path: str, epoch: int):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_exists = os.path.exists(log_path)

    if not log_exists:
        with open(log_path, 'w') as f:
            header = ['epoch'] + list(metrics.keys())
            f.write(','.join(header) + '\n')

    with open(log_path, 'a') as f:
        values = [f"{epoch:03d}"] + [f"{v:.6f}" if isinstance(v, float) else str(v) for v in metrics.values()]
        f.write(','.join(values) + '\n')


def gradient_loss(x, y):
    dx_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx_y = y[:, :, 1:, :] - y[:, :, :-1, :]
    dy_y = y[:, :, :, 1:] - y[:, :, :, :-1]

    return F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)


def compute_recon_loss(images, recons, lambda_edge=0.1):
    l1 = F.l1_loss(recons, images)
    edge = gradient_loss(recons, images)
    return l1 + lambda_edge * edge


def create_scheduler(optimizer, scheduler_name, warmup_steps, total_steps, start_epoch):
    if scheduler_name == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, last_epoch=start_epoch)
    
    if scheduler_name == 'constant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, last_epoch=start_epoch)
    
    if scheduler_name == 'linear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, last_epoch=start_epoch)
    
    raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")
