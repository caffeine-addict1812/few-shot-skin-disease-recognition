"""Download SD-198 from the HuggingFace mirror, pin the revision, hash the zip, extract.

    python scripts/download_data.py                # -> data/raw/sd-198.zip, data/SD-198-raw/, data/download_info.json

The mirror is a single zip, so provenance is recorded as repo + revision SHA + zip sha256/md5.
"""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "resyhgerwshshgdfghsdfgh/SD-198"
# Revision inspected on 2026-10-03; pinned so every machine gets identical bytes.
REVISION = "68c02939c06d6850f40f2911e1d1d49bd760139d"
ZIP_NAME = "sd-198.zip"


def file_hashes(path, chunk=1 << 22):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--extract-dir", default="data/SD-198-raw")
    ap.add_argument("--info", default="data/download_info.json")
    args = ap.parse_args()

    raw_dir, extract_dir, info_path = Path(args.raw_dir), Path(args.extract_dir), Path(args.info)
    raw_dir.mkdir(parents=True, exist_ok=True)
    info_path.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    ds = api.dataset_info(args.repo, revision=args.revision, files_metadata=True)
    assert ds.sha == args.revision, f"resolved {ds.sha}, expected {args.revision}"
    print(f"{args.repo} @ {ds.sha} (gated={ds.gated})")
    expected_sha256 = next(s.lfs.sha256 for s in ds.siblings if s.rfilename == ZIP_NAME)

    # A zip already in raw_dir (e.g. finished by hand with `curl -C -` after an interrupted
    # download) is accepted only if it matches the LFS sha256 published for this revision.
    zip_path = raw_dir / ZIP_NAME
    if zip_path.exists():
        print(f"{zip_path} exists, verifying instead of re-downloading")
    else:
        zip_path = Path(hf_hub_download(args.repo, ZIP_NAME, repo_type="dataset",
                                        revision=args.revision, local_dir=raw_dir))
    print("zip:", zip_path, f"{zip_path.stat().st_size / 2**30:.3f} GiB")

    sha256, md5 = file_hashes(zip_path)
    print("sha256:", sha256, "\nmd5:   ", md5)
    assert sha256 == expected_sha256, f"sha256 mismatch: got {sha256}, HF LFS says {expected_sha256}"
    print("sha256 matches the HF LFS hash for this revision")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"extracting {len(names)} entries -> {extract_dir}")
            zf.extractall(extract_dir)
        else:
            print(f"{extract_dir} already populated, skipping extract")

    info = {
        "repo": args.repo,
        "revision": args.revision,
        "zip_name": ZIP_NAME,
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256,
        "zip_md5": md5,
        "zip_entries": len(names),
        "license": "other (HF card page lists 'unknown'); original: Sun et al., ECCV 2016",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print("wrote", info_path)


if __name__ == "__main__":
    main()
