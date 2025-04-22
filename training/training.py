import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib
matplotlib.use("Agg")

import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

from src.ddpm import DDPMSampler
from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from src.diffusion import Diffusion
from utils.dataset import EmojiDataset


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
    save_path = os.path.join(save_dir, f'epoch_{epoch:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
    plt.close()


def train_step(batch, models, scheduler, optimizer, device):
    encoder, decoder, diffusion, text_encoder = models['encoder'], models['decoder'], models['diffusion'], models['text_encoder']

    images = batch['image'].to(device)
    tokens = batch['tokens'].to(device)
    context = text_encoder(tokens).last_hidden_state

    # === Encode images ===
    with torch.no_grad():
        noise = torch.randn(images.size(0), 4, 32, 32).to(device)
        latents = encoder(images, noise)

    # === Diffusion process ===
    bsz = latents.size(0)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.num_training_steps, (bsz,), device=device)
    noisy_latents = scheduler.add_noise(latents, timesteps)

    # === Predict noise ===
    pred_noise = diffusion(noisy_latents, context, timesteps)

    # === Loss ===
    loss = F.mse_loss(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(model_dict, dataloader, optimizer, scheduler, device, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        pbar = tqdm(dataloader)

        for batch in pbar:
            loss = train_step(batch, model_dict, scheduler, optimizer, device)
            pbar.set_postfix({"loss": loss})
        
        sample_and_log(
            diffusion=model_dict['diffusion'],
            decoder=model_dict['decoder'],
            tokenizer=tokenizer,
            text_encoder=model_dict['text_encoder'],
            scheduler=scheduler,
            device=device,
            epoch=epoch
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_path = project_root / 'data' / 'tokenizer'

tokenizer = CLIPTokenizer(
    vocab_file=str(tokenizer_path / 'vocab.json'),
    merges_file=str(tokenizer_path / 'merges.txt'),
)

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
encoder = VAE_Encoder().to(device)
decoder = VAE_Decoder().to(device)
diffusion = Diffusion().to(device)

dataset = EmojiDataset(str(project_root / 'data/emoji_dataset_128x128/emoji_dataset.json'), image_size=128, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

generator = torch.Generator(device=device)

seed = 42

if seed is None:
    generator.seed()
else:
    generator.manual_seed(seed)
    
scheduler = DDPMSampler(generator)
scheduler.set_inference_timesteps(num_inference_steps=50)

optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-4)

models = {
    'encoder': encoder,
    'decoder': decoder,
    'diffusion': diffusion,
    'text_encoder': text_encoder
}

train(models, dataloader, optimizer, scheduler, device, epochs=1)
