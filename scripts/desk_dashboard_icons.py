#!/usr/bin/env python3
"""Generate PWA / Apple touch icons for Desk 看板."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "#141415")
    draw = ImageDraw.Draw(img)
    text = "看板"
    f = font(int(size * 0.36))
    bbox = draw.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1]
    draw.text((x, y), text, font=f, fill="#ededed")
    return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".", help="output directory")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    icon = make_icon(512)
    icon.save(out / "icon-512.png", "PNG")
    icon.resize((192, 192), Image.Resampling.LANCZOS).save(out / "icon-192.png", "PNG")
    icon.resize((180, 180), Image.Resampling.LANCZOS).save(out / "apple-touch-icon.png", "PNG")
    print(f"wrote icons to {out.resolve()}")


if __name__ == "__main__":
    main()
