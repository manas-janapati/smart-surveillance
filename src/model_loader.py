import torch
import torch.nn as nn
import timm

NUM_CLASSES = 6

class BEiTWithXP5Head(nn.Module):
    def __init__(self):
        super().__init__()

        # BEiT backbone WITHOUT classifier
        self.backbone = timm.create_model(
            "beit_base_patch16_384",
            pretrained=False,
            num_classes=0
        )

        self.layer_norm = nn.LayerNorm(self.backbone.num_features)
        self.classifier = nn.Linear(self.backbone.num_features, NUM_CLASSES)

    def forward(self, x):
        features = self.backbone(x)   # (B, D)
        features = self.layer_norm(features)
        return self.classifier(features)


def load_model(beit_path, xp5_head_path, device="cpu"):
    model = BEiTWithXP5Head().to(device)

    # -------- Load BEiT backbone weights (REMOVE HEAD) --------
    beit_ckpt = torch.load(beit_path, map_location=device)

    # Remove original head weights if present
    beit_ckpt = {
        k: v for k, v in beit_ckpt.items()
        if not k.startswith("head.")
    }

    model.backbone.load_state_dict(beit_ckpt, strict=True)

    # -------- Load XP5 classifier head --------
    xp5_ckpt = torch.load(xp5_head_path, map_location=device)

    model.layer_norm.load_state_dict({
        "weight": xp5_ckpt["layer_norm.weight"],
        "bias": xp5_ckpt["layer_norm.bias"],
    })

    model.classifier.load_state_dict({
        "weight": xp5_ckpt["linear_classif.weight"],
        "bias": xp5_ckpt["linear_classif.bias"],
    })

    model.eval()
    return model
