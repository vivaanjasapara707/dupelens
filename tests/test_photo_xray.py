"""Photo X-ray on made-up photos: the one product that looks different gets the highest odd-one-out score."""
import numpy as np
from PIL import Image

from src.photo_xray import cutout, xray


def product(rgb, shift=0):
    a = np.full((80, 80, 3), 255, dtype=np.uint8)
    a[20 + shift:60 + shift, 25:55] = rgb
    return Image.fromarray(a)


def test_cutout_removes_white_background():
    assert cutout(product((200, 30, 30))).shape == (48, 48, 3)


def test_odd_photo_is_flagged():
    ims = [product((200, 30, 30), shift=k % 3) for k in range(6)] + [product((30, 60, 210))]
    out = xray(ims)
    z = [it["z"] for it in out["items"]]
    assert int(np.argmax(z)) == 6 and out["items"][6]["odd"] and len(out["eigen"]) == 3
