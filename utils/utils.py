import io
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image
from IPython.display import display, Image as IPyImage


def tensor_image_grid(tensor, title=None, prompts=None, vae_only=False, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tensor = (tensor.clamp(-1, 1) + 1) / 2
    grid = vutils.make_grid(tensor, nrow=len(tensor) if tensor.size(0) <= 8 else 8)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(grid_image.shape[1] // 50, grid_image.shape[0] // 50))
    ax.imshow(grid_image, interpolation='nearest')
    ax.axis('off')

    if vae_only:
        ax.text(0, 0, 'VAE_ONLY', color='white', fontsize=12, backgroundcolor='black')

    if prompts:
        width = grid_image.shape[1] // len(prompts)
        for i, prompt in enumerate(prompts):
            ax.text(i * width + width // 2, grid_image.shape[0] + 5, prompt, fontsize=10, ha='center', va='top')

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
    else:
        plt.show()

    plt.close()


@torch.no_grad()
def sample_and_log(diffusion, decoder, tokenizer, text_encoder, scheduler, device, epoch, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    diffusion.eval()
    decoder.eval()

    prompts = ['happy cat', 'crying face', 'robot with heart eyes', 'surprised ghost']
    all_images = []

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=77, truncation=True)['input_ids'].to(device)
        context_out = text_encoder(tokens)
        context = context_out.last_hidden_state if hasattr(context_out, 'last_hidden_state') else context_out

        latents = torch.randn(1, 4, 32, 32).to(device)

        for t in scheduler.timesteps:
            time = torch.tensor([t], device=device)
            noise_pred = diffusion(latents, context, time)
            latents = scheduler.step(time[0], latents, noise_pred)

        image = decoder(latents)
        all_images.append(image)

    all_images = torch.cat(all_images)

    if save:
        save_path = os.path.join(save_path, f'epoch_{epoch+1:04d}.png')

    tensor_image_grid(all_images, prompts=prompts, save=save, save_path=save_path)


@torch.no_grad()
def log_reconstructions(encoder, decoder, dataloader, device, epoch, vae_only=False, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    encoder.eval()
    decoder.eval()

    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    noise = torch.randn(images.size(0), 4, 32, 32).to(device)
    latents = encoder(images, noise)
    recon_images = decoder(latents)

    combined = torch.cat([images, recon_images], dim=0)

    if save:
        if vae_only:
            save_path = os.path.join(save_path, f"vae_recon_epoch_{epoch:04d}.png")
        else:
            save_path = os.path.join(save_path, f"epoch_{epoch+1:04d}_vae_recons.png")

    tensor_image_grid(combined, title='VAE Reconstructions', vae_only=vae_only, save=save, save_path=save_path)


@torch.no_grad()
def sample_from_vae(decoder, device, num_samples=8, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    decoder.eval()
    latents = torch.randn(num_samples, 4, 32, 32).to(device)
    images = decoder(latents)

    tensor_image_grid(images, title='Random VAE Samples', vae_only=True, save=save, save_path=save_path)


@torch.no_grad()
def interpolate_vae(decoder, device, steps=8, epoch=None, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    decoder.eval()

    z_start = torch.randn(1, 4, 32, 32).to(device)
    z_end = torch.randn(1, 4, 32, 32).to(device)

    interpolated = torch.cat([
        (1 - alpha) * z_start + alpha * z_end
        for alpha in torch.linspace(0, 1, steps)
    ])

    images = decoder(interpolated)

    if save:
        filename = f"interpolation_epoch_{epoch:04d}.png" if epoch is not None else 'interpolation.png'
        save_path = os.path.join(save_path, filename)

    tensor_image_grid(images, title='VAE Latent Interpolation', vae_only=True, save=save, save_path=save_path)


@torch.no_grad()
def interpolate_between_images(encoder, decoder, dataset, img1_id, img2_id, device, steps=8, epoch=None, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    encoder.eval()
    decoder.eval()

    img1 = dataset[img1_id]['image'].unsqueeze(0).to(device)
    img2 = dataset[img2_id]['image'].unsqueeze(0).to(device)

    noise = torch.randn(2, 4, 32, 32).to(device)
    z1 = encoder(img1, noise[0:1])
    z2 = encoder(img2, noise[1:2])

    alphas = torch.linspace(0, 1, steps).to(device)

    interpolated = torch.cat([
        ((1 - alpha) * z1 + alpha * z2)
        for alpha in alphas
    ], dim=0)

    decoded = decoder(interpolated)

    if save:
        filename = f"real_interpolation_epoch_{epoch:04d}.png" if epoch is not None else 'real_interpolation.png'
        save_path = os.path.join(save_path, filename)

    tensor_image_grid(decoded, title='VAE Interpolation Between Real Images', vae_only=True, save=save, save_path=save_path)


@torch.no_grad()
def interpolate_to_gif(encoder, decoder, dataset, img1_id, img2_id, device, steps=8, epoch=None, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    encoder.eval()
    decoder.eval()

    img1 = dataset[img1_id]['image'].unsqueeze(0).to(device)
    img2 = dataset[img2_id]['image'].unsqueeze(0).to(device)

    noise = torch.randn(1, 4, 32, 32).to(device)
    z1 = encoder(img1, noise)
    z2 = encoder(img2, noise)

    alphas = torch.linspace(0, 1, steps).to(device)
    latents = torch.stack([(1 - a) * z1 + a * z2 for a in alphas], dim=0).squeeze(1)

    decoded = decoder(latents)
    decoded = (decoded.clamp(-1, 1) + 1) / 2

    frames = [to_pil_image(img.cpu()) for img in decoded]

    total_duration_ms = 2500
    frame_duration = max(50, total_duration_ms // steps)

    if save:
        filename = f"interpolation_epoch_{epoch:04d}.gif" if epoch is not None else "interpolation.gif"
        gif_path = os.path.join(save_path, filename)

    if save:
        frames[0].save(
            gif_path,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=frame_duration,
            loop=0
        )
    else:
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=frame_duration,
            loop=0
        )
        buf.seek(0)
        display(IPyImage(data=buf.getvalue()))


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
