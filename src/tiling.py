from PIL import Image


def tile_image(image, tile_size=384, stride=384):
    w, h = image.size
    tiles = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append(tile)

    return tiles
