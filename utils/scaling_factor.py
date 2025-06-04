import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.dataset import StickerDataset
from src.encoder import VAE_Encoder


JSON_PATH = "data/sticker_dataset_128x128/dataset.json"
IMAGE_DIR = "data/sticker_dataset_128x128/images"
BLACKLIST = "data/sticker_dataset_128x128/blacklist.txt"
VAE_CKPT = "checkpoints/vae/vae-1.0/vae_epoch_0100.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64


dataset = StickerDataset(JSON_PATH, image_dir=IMAGE_DIR, tokenize=False, blacklist=BLACKLIST)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


encoder = VAE_Encoder().to(DEVICE)
encoder.eval()
if VAE_CKPT:
    print("Loading VAE checkpoint...")
    ckpt = torch.load(VAE_CKPT, map_location='cpu')
    encoder.load_state_dict(ckpt['encoder'])


all_mu = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Encoding images"):
        images = batch['image'].to(DEVICE)
        mu, _ = encoder(images)
        all_mu.append(mu.cpu())


all_mu = torch.cat(all_mu, dim=0)
mu_flat = all_mu.flatten().numpy()

std = np.std(mu_flat)
scaling = 1.0 / std

print(f"Latent std: {std:.5f}")
print(f"Scaling factor (1/std): {scaling:.5f}")
