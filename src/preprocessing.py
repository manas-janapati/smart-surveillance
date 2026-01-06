import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


def preprocess_tiles(tiles):
    return torch.stack([transform(tile) for tile in tiles])
