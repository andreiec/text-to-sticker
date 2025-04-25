import io
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image
from IPython.display import display, Image as IPyImage


def tensor_image_grid(tensor, title=None, prompts=None, save=False, save_path=''):
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
def sample_and_log(diffusion, decoder, tokenizer, text_encoder, scheduler, device, epoch: int, save: bool = False, save_path: str = ''):
    if save and not save_path:
        raise ValueError("Must provide a save_path when save=True")

    diffusion.eval()
    decoder.eval()
    text_encoder.eval()

    prompts = ['happy cat', 'crying face', 'robot with heart eyes', 'surprised ghost']
    all_images = []

    timesteps = scheduler.inference_timesteps.to(device)

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=77, truncation=True).input_ids.to(device)
        context = text_encoder(tokens).last_hidden_state
        latents = torch.randn(1, 4, 32, 32, device=device)

        for idx, t in enumerate(timesteps):
            t_int = int(t)
            t_tensor = torch.tensor([t_int], device=device)

            noise_pred = diffusion(latents, context, t_tensor)
            latents = scheduler.step(latents, noise_pred, t_int, idx)

        image = decoder(latents)
        all_images.append(image)

    all_images = torch.cat(all_images, dim=0)

    if save:
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, f'epoch_{epoch+1:04d}.png')

    tensor_image_grid(all_images, prompts=prompts, save=save, save_path=out_file)


@torch.no_grad()
def log_reconstructions(encoder, decoder, dataloader, device, epoch, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    encoder.eval()
    decoder.eval()

    batch = next(iter(dataloader))
    images = batch['image'].to(device)

    mu, _ = encoder(images)
    latents = mu
    recon_images = decoder(latents)

    combined = torch.cat([images, recon_images], dim=0)

    if save:
        save_path = os.path.join(save_path, f"epoch_{epoch+1:04d}_recons.png")

    tensor_image_grid(combined, title='VAE Reconstructions', save=save, save_path=save_path)


@torch.no_grad()
def log_reconstructions_vae(encoder, decoder, dataloader, device, epoch, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    encoder.eval()
    decoder.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    mu, logvar = encoder(images)
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    z = mu + eps * std

    recons = decoder(z)

    stacked = torch.cat([images, recons])

    if save:
        save_path = os.path.join(save_path, f"epoch_{epoch+1:04d}_vae_recons.png")

    tensor_image_grid(stacked, title='VAE Reconstructions', save=save, save_path=save_path)


@torch.no_grad()
def sample_from_vae(decoder, device, num_samples=8, save=False, save_path=''):
    if save and save_path == '':
        raise ValueError('Invalid save path')

    decoder.eval()
    latents = torch.randn(num_samples, 4, 32, 32).to(device)
    images = decoder(latents)
    tensor_image_grid(images, title='Random VAE Samples', save=save, save_path=save_path)


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

    tensor_image_grid(images, title='VAE Latent Interpolation', save=save, save_path=save_path)


@torch.no_grad()
def interpolate_between_images(encoder, decoder, dataset, img1_id, img2_id, device, steps=8, epoch=None, save=False, save_path=''):
    encoder.eval()
    decoder.eval()

    img1 = dataset[img1_id]['image'].unsqueeze(0).to(device)
    img2 = dataset[img2_id]['image'].unsqueeze(0).to(device)

    mu1, logvar1 = encoder(img1)
    mu2, logvar2 = encoder(img2)

    z1 = mu1 + torch.randn_like(mu1) * (0.5 * logvar1).exp()
    z2 = mu2 + torch.randn_like(mu2) * (0.5 * logvar2).exp()

    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated = torch.stack([(1 - a) * z1 + a * z2 for a in alphas], dim=0).squeeze(1)

    decoded = decoder(interpolated)

    if save:
        filename = f"real_interpolation_epoch_{epoch:04d}.png" if epoch is not None else 'real_interpolation.png'
        save_path = os.path.join(save_path, filename)

    tensor_image_grid(decoded, title='VAE Interpolation Between Real Images', save=save, save_path=save_path)


@torch.no_grad()
def interpolate_to_gif(encoder, decoder, dataset, img1_id, img2_id, device, steps=8, epoch=None, save=False, save_path=''):
    encoder.eval()
    decoder.eval()

    img1 = dataset[img1_id]['image'].unsqueeze(0).to(device)
    img2 = dataset[img2_id]['image'].unsqueeze(0).to(device)

    mu1, logvar1 = encoder(img1)
    mu2, logvar2 = encoder(img2)

    z1 = mu1 + torch.randn_like(mu1) * (0.5 * logvar1).exp()
    z2 = mu2 + torch.randn_like(mu2) * (0.5 * logvar2).exp()

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
