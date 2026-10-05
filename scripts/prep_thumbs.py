"""Resize every image so its shorter side is 256 px (what Resize(256) would do) and store as JPEG q95.

    python scripts/prep_thumbs.py --src data/SD-198-raw --dst data/SD-198-256

Class = name of the image's parent folder, so nested zip layouts are fine. Originals reach 4750 px;
decoding those every episode would make the dataloader the bottleneck, so this step is not optional.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def convert(job):
    src, dst, size = job
    if dst.exists() and dst.stat().st_size > 0:
        return "skipped"
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")  # phone photos carry EXIF rotation
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.BICUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=95)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/SD-198-raw")
    ap.add_argument("--dst", default="data/SD-198-256")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(p for p in src.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXT
                   and not p.name.startswith(".") and "__MACOSX" not in p.parts)
    jobs, seen = [], {}
    for p in files:
        out = dst / p.parent.name / (p.stem + ".jpg")
        if out in seen:
            raise SystemExit(f"name collision after conversion: {p} vs {seen[out]}")
        seen[out] = p
        jobs.append((p, out, args.size))
    print(f"{len(jobs)} images in {len({j[1].parent for j in jobs})} classes -> {dst}")

    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(convert, jobs, chunksize=16))
    print({r: results.count(r) for r in set(results)})


if __name__ == "__main__":
    main()
