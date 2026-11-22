"""Create canonical API copies of all benchmark images (used identically by EVERY condition: captioning,
embedding, reading). Rule: downscale only if max side > 2048 px (LANCZOS); PNG kept as PNG if <= 1.5 MB after
resizing, otherwise JPEG quality 95; JPEG/WEBP -> JPEG q95. Gemini 3 'high' media resolution (~1.1k tokens/image)
processes far fewer pixels than 2048 px, so this should not remove information the model would see.
Writes data/_api_images/... and data/_api_images/manifest.json (original -> canonical).
"""
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/_api_images"
SRC = [ROOT / "data/MemEye/data/image", ROOT / "data/Mem-Gallery/data/image"]
MAX_SIDE = 2048


def canon(src: Path) -> Path:
    rel = src.relative_to(ROOT / "data")
    im = Image.open(src)
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        im = im.convert("RGB" if im.mode not in ("RGB", "RGBA", "L") else im.mode).resize(
            (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    if src.suffix.lower() == ".png":
        b = io.BytesIO()
        im.save(b, format="PNG", optimize=True)
        if b.tell() <= 1.5e6:
            dst = OUT / rel.with_suffix(".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b.getvalue())
            return dst
    dst = OUT / rel.with_suffix(".jpg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(dst, format="JPEG", quality=95)
    return dst


def main():
    man = {}
    for root in SRC:
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p.is_file():
                d = canon(p)
                man[str(p)] = {"canon": str(d), "orig_mb": round(p.stat().st_size / 1e6, 3),
                               "canon_mb": round(d.stat().st_size / 1e6, 3), "orig_size": Image.open(p).size,
                               "canon_size": Image.open(d).size}
    json.dump(man, open(OUT / "manifest.json", "w"), indent=0)
    tot_o = sum(v["orig_mb"] for v in man.values())
    tot_c = sum(v["canon_mb"] for v in man.values())
    print(f"{len(man)} images: {tot_o:.0f} MB -> {tot_c:.0f} MB; resized: "
          f"{sum(v['orig_size'] != v['canon_size'] for v in man.values())}")


if __name__ == "__main__":
    sys.exit(main())
