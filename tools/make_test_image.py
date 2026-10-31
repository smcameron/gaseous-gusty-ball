#!/usr/bin/env python3
"""Generate a tall, thin banded test image for gaseous-gusty-ball (needs Pillow).

usage: make_test_image.py out.png [width] [height] [seed]
"""
import math
import random
import sys

from PIL import Image


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "bands.png"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    rnd = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 1)

    palette = [(222, 196, 160), (190, 140, 100), (240, 228, 205), (160, 100, 70),
               (210, 170, 120), (120, 80, 60), (235, 215, 180), (180, 120, 85)]
    # random band boundaries
    nb = 22
    cuts = sorted(rnd.random() for _ in range(nb - 1))
    edges = [0.0] + cuts + [1.0]
    cols = [palette[rnd.randrange(len(palette))] for _ in range(nb)]

    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = (y + 0.5) / h
        k = 0
        while k < nb - 1 and edges[k + 1] < t:
            k += 1
        c0 = cols[k]
        # soften band edges
        c1 = cols[min(k + 1, nb - 1)]
        f = (t - edges[k]) / max(1e-9, edges[k + 1] - edges[k])
        s = max(0.0, (f - 0.85) / 0.15)
        base = [c0[i] * (1 - s) + c1[i] * s for i in range(3)]
        for x in range(w):
            n = 12 * math.sin(2 * math.pi * x / w * 3 + y * 0.05) + rnd.uniform(-10, 10)
            px[x, y] = tuple(int(max(0, min(255, base[i] + n))) for i in range(3))
    img.save(out)
    print(f"wrote {out} ({w}x{h})")


if __name__ == "__main__":
    main()
