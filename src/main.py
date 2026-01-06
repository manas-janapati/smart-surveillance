import os

# ---------------- CPU OPTIMIZATION ----------------
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.mkldnn.enabled = True
torch.backends.openmp.enabled = True
# -------------------------------------------------

from PIL import Image
from tqdm import tqdm

from model_loader import load_model
from tiling import tile_image
from preprocessing import preprocess_tiles
from inference import predict_image


IMAGE_DIR = "samples/test/."
BACKBONE_PATH = "models/model_beit_danish.pth"
HEAD_PATH = "models/model_tiling_state_dict_xp5.pth"
THRESHOLD = 0.5
TILE_BATCH_SIZE = 8


def short_name(filename, max_len=50):
    return filename if len(filename) <= max_len else filename[:47] + "..."


def main():
    device = "cpu"
    model = load_model(BACKBONE_PATH, HEAD_PATH, device)

    images = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    invasive_count = 0

    for img_name in tqdm(images, desc="Processing", leave=False):
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = Image.open(img_path).convert("RGB")

        # Fast downscale gate
        if image.width > 768:
            new_h = int(768 * image.height / image.width)
            image = image.resize((768, new_h))

        tiles = tile_image(image)
        if not tiles:
            continue

        tile_tensor = preprocess_tiles(tiles)

        invasive, _ = predict_image(
            model,
            tile_tensor,
            threshold=THRESHOLD,
            batch_size=TILE_BATCH_SIZE
        )

        label = "INVASIVE" if invasive else "OK"
        if invasive:
            invasive_count += 1

        print(f"{label:<9} | {short_name(img_name)}")

    print("\nSummary")
    print("-------")
    print(f"Total images     : {len(images)}")
    print(f"Invasive detected: {invasive_count}")


if __name__ == "__main__":
    main()
