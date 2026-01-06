import torch


@torch.no_grad()
def predict_image(
    model,
    tile_tensor,
    threshold=0.5,
    batch_size=8
):
    """
    Batched + early-exit inference
    """
    max_conf = 0.0

    for i in range(0, tile_tensor.size(0), batch_size):
        batch = tile_tensor[i:i + batch_size]
        logits = model(batch)
        probs = torch.sigmoid(logits)

        batch_max = probs.max().item()
        max_conf = max(max_conf, batch_max)

        # EARLY EXIT (huge speedup)
        if max_conf >= threshold:
            return True, max_conf

    return False, max_conf
