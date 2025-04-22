import json
import torchvision.transforms as T

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EmojiDataset(Dataset):
    def __init__(self, json_path, image_size=128, tokenize=True, tokenizer=None, max_length=77):
        self.json_path = Path(json_path)
        self.root_dir = self.json_path.parent
        self.tokenize = tokenize
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        excluded_descriptions = {
            'rainbow flag',
            'transgender flag',
            'pirate flag'
        }

        # Filter out country flags
        self.data = [
            item for item in self.data
            if not item.get('description', '').lower().startswith('flag: ')
            and item.get('description', '').lower() not in excluded_descriptions
        ]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.root_dir / Path(item['image_path'].replace('\\', '/'))
        description = item.get('description', '')
        keywords_raw = item.get('keywords', '')

        if isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(',')]
        else:
            keywords = keywords_raw

        text_input = description + ' ' + ' '.join(keywords)

        image = Image.open(image_path).convert('RGBA').convert('RGB')
        image = self.transform(image)
        
        sample = {'image': image}

        if self.tokenize:
            if not self.tokenizer:
                raise ValueError('No tokenizer specified')

            tokens = self.tokenizer(
                text_input,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze(0)

            sample['tokens'] = tokens
            sample['description'] = text_input

        return sample

