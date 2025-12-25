import torch

def predict_probs(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0)
