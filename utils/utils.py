import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from PIL import Image


@torch.no_grad()
def sample_and_log(diffusion, decoder, tokenizer, text_encoder, scheduler, device, epoch, save_dir='samples'):
    diffusion.eval()
    decoder.eval()

    prompts = [
        "happy cat",
        "crying face",
        "robot with heart eyes",
        "surprised ghost"
    ]

    os.makedirs(save_dir, exist_ok=True)

    all_images = []

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=77, truncation=True)["input_ids"].to(device)
        context_out = text_encoder(tokens)
        context = context_out.last_hidden_state if hasattr(context_out, "last_hidden_state") else context_out

        latents = torch.randn(1, 4, 32, 32).to(device)

        for t in scheduler.timesteps:
            time = torch.tensor([t], device=device)
            noise_pred = diffusion(latents, context, time)
            latents = scheduler.step(time[0], latents, noise_pred)

        image = decoder(latents)
        image = (image.clamp(-1, 1) + 1) / 2
        all_images.append(image)

    grid = vutils.make_grid(torch.cat(all_images), nrow=len(prompts))
    grid_image = grid.permute(1, 2, 0).cpu().numpy()

    # --- Plot with labels ---
    fig, ax = plt.subplots(figsize=(len(prompts) * 2.5, 3))
    ax.imshow(grid_image)
    ax.axis('off')

    # Add text labels
    width = grid_image.shape[1] // len(prompts)
    for i, prompt in enumerate(prompts):
        ax.text(i * width + width // 2, grid_image.shape[0] + 5, prompt,
                fontsize=10, ha='center', va='top', color='black')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
    plt.close()


@torch.no_grad()
def log_vae_reconstructions(encoder, decoder, dataloader, device, epoch, save_dir="vae_recons"):
    os.makedirs(save_dir, exist_ok=True)
    encoder.eval()
    decoder.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    noise = torch.randn(images.size(0), 4, 32, 32).to(device)
    latents = encoder(images, noise)
    recon_images = decoder(latents)

    images = (images + 1) / 2
    recon_images = (recon_images + 1) / 2

    stacked = torch.cat([images, recon_images])

    grid = vutils.make_grid(stacked, nrow=images.size(0))
    ndarr = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(ndarr)
    img.save(os.path.join(save_dir, f"epoch_{epoch+1:04d}_vae_recons.png"))
