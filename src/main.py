import torch
from PIL import Image

from model_loader import load_model
from preprocessing import get_transform
from inference import predict_probs

# -------- PATHS --------
BEIT_PATH = "models/model_beit_danish.pth"
XP5_PATH = "models/model_tiling_state_dict_xp5.pth"
IMAGE_PATH = "samples/test_camara.jpg"

# -------- CLASS LABELS (XP5 = invasive species only) --------
CLASSES = [
    "Solidago",
    "Cytisus_scoparius",
    "Rosa_rugosa",
    "Lupinus_polyphyllus",
    "Pastinaca_sativa",
    "Reynoutria"
]

# -------- SURVEILLANCE PARAMETERS --------
INVASIVE_THRESHOLD = 0.4      # total invasive confidence
SPECIES_MIN_CONF = 0.15       # minimum confidence to name species


def main():
    print("Loading model...")
    model = load_model(BEIT_PATH, XP5_PATH)

    image = Image.open(IMAGE_PATH).convert("RGB")
    tensor = get_transform()(image)

    probs = predict_probs(model, tensor)

    # -------- RAW OUTPUT --------
    print("\nRaw invasive species probabilities:")
    for cls, p in zip(CLASSES, probs):
        print(f"{cls:25s}: {p:.4f}")

    # -------- AGGREGATION --------
    invasive_score = probs.sum().item()
    non_invasive_score = 1.0 - invasive_score

    # -------- TOP SPECIES --------
    top_idx = torch.argmax(probs).item()
    top_species = CLASSES[top_idx]
    top_species_conf = probs[top_idx].item()

    print("\nScores:")
    print("Invasive Score     :", round(invasive_score, 4))
    print("Non-Invasive Score :", round(non_invasive_score, 4))

    # -------- DECISION LOGIC --------
    if invasive_score > INVASIVE_THRESHOLD:
        print("\nINVASIVE PLANT DETECTED")

        if top_species_conf > SPECIES_MIN_CONF:
            print("Detected Species   :", top_species)
            print("Species Confidence :", round(top_species_conf, 4))

        print("Final Invasive Score:", round(invasive_score, 4))
    else:
        print("\nNON-INVASIVE / BACKGROUND VEGETATION")
        print("Confidence         :", round(non_invasive_score, 4))


if __name__ == "__main__":
    main()



"""
import torch
from PIL import Image

from model_loader import load_model
from preprocessing import get_transform
from inference import predict_probs

# -------- PATHS --------
BEIT_PATH = "models/model_beit_danish.pth"
XP5_PATH = "models/model_tiling_state_dict_xp5.pth"
IMAGE_PATH = "samples/invasive_1.jpg"

# -------- SURVEILLANCE PARAMETERS --------
INVASIVE_THRESHOLD = 0.35   # tune between 0.3 – 0.6


def main():
    print("Loading model...")
    model = load_model(BEIT_PATH, XP5_PATH)

    image = Image.open(IMAGE_PATH).convert("RGB")
    tensor = get_transform()(image)

    probs = predict_probs(model, tensor)

    # -------- RAW OUTPUT --------
    print("\nRaw invasive class probabilities:")
    for i, p in enumerate(probs):
        print(f"Class {i+1}: {p:.4f}")

    # -------- DECISION LOGIC --------
    max_invasive_prob = probs.max().item()

    print("\nMax invasive confidence:", round(max_invasive_prob, 4))

    if max_invasive_prob > INVASIVE_THRESHOLD:
        print("\nINVASIVE PLANT DETECTED")
        print("Confidence:", round(max_invasive_prob, 4))
    else:
        print("\nNON-INVASIVE / BACKGROUND VEGETATION")
        print("Confidence:", round(1.0 - max_invasive_prob, 4))


if __name__ == "__main__":
    main()


"""