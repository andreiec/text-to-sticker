import os
import math
import torch


def save_checkpoint(models, optimizer, epoch, path, vae_only=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        'encoder': models['encoder'].state_dict(),
        'decoder': models['decoder'].state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    if not vae_only:
        state['diffusion'] = models['diffusion'].state_dict()

    torch.save(state, path)


def load_checkpoint(models, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    models['encoder'].load_state_dict(checkpoint['encoder'])
    models['decoder'].load_state_dict(checkpoint['decoder'])

    if 'diffusion' in checkpoint and 'diffusion' in models:
        models['diffusion'].load_state_dict(checkpoint['diffusion'])

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


import os

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
