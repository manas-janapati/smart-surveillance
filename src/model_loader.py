import torch
import torch.nn as nn
import timm


class XP5Head(nn.Module):
    def __init__(self, embed_dim, num_classes=6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear_classif = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear_classif(x)
        return x


def load_model(backbone_path, head_path, device):
    backbone = timm.create_model(
        "beit_base_patch16_384",
        pretrained=False,
        num_classes=0
    )

    backbone.load_state_dict(
        torch.load(backbone_path, map_location="cpu"),
        strict=False
    )

    head = XP5Head(backbone.num_features, 6)
    head.load_state_dict(
        torch.load(head_path, map_location="cpu"),
        strict=True
    )

    model = nn.Sequential(backbone, head)
    model.to(device)
    model.eval()

    return model
