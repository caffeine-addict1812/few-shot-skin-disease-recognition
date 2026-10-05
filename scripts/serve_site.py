"""Serve the project site on localhost, with the trained model behind it for the demo's "Live model" mode.

    python scripts/serve_site.py                 # http://127.0.0.1:8765
    python scripts/serve_site.py --port 9000

Routes
  GET  /...                 static files from site/
  GET  /img/<path>          a photo from data/SD-198-256 (for hover previews of training photos)
  GET  /api/health          whether the live model is loaded
  POST /api/embed           body = image bytes (an uploaded photo), or ?path=<dataset path> to re-embed a dataset photo.
                            Runs both models (frozen ImageNet ResNet-18 and the fine-tuned checkpoint) exactly as
                            src/model.py does at test time, and returns both 512-d embeddings, a 3-D position in the
                            galaxy (inverse-distance mean of its 10 nearest photos in 512-d; t-SNE has no exact
                            out-of-sample map) and the nearest photos. Research demo: not a diagnostic tool.
Standard library only (no web framework); binds to 127.0.0.1 by default.
"""
import argparse
import io
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import build_transforms  # noqa: E402
from model import ProtoNet, load_model  # noqa: E402

SITE, DATA = ROOT / "site", ROOT / "data/SD-198-256"
MAX_UPLOAD = 15 * 1024 * 1024
K_PLACE = 10

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
TF = build_transforms(False)
MODELS = {"finetuned": load_model(ROOT / "checkpoints/best.pt", "cpu")[0], "frozen": ProtoNet(pretrained=True).eval()}
C = np.load(ROOT / "results/site_cache.npz")
EMB = {m: C[f"emb_{m}"].astype(np.float32) for m in MODELS}
XYZ = {m: C[f"xyz_{m}"].astype(np.float32) for m in MODELS}
CLS, SPLIT, NAMES, PATHS = C["cls"], C["split"], C["names"], C["paths"]
PATH_INDEX = {str(p): i for i, p in enumerate(PATHS)}
LOCK = threading.Lock()


def embed(img: Image.Image):
    x = TF(ImageOps.exif_transpose(img).convert("RGB")).unsqueeze(0)
    with LOCK, torch.no_grad():
        return {m: mod.embed(x)[0].numpy().astype(np.float32) for m, mod in MODELS.items()}


def describe(z, self_index=None):
    out = {}
    for m in MODELS:
        d2 = ((EMB[m] - z[m]) ** 2).sum(1)
        near = np.argsort(d2)[:K_PLACE]
        w = 1.0 / (d2[near] + 1e-4)
        out[m] = dict(embedding=[round(float(v), 6) for v in z[m]],
                      xyz=[round(float(v), 5) for v in (XYZ[m][near] * w[:, None]).sum(0) / w.sum()],
                      neighbours=[dict(index=int(i), name=str(NAMES[CLS[i]]), split=int(SPLIT[i]), path=str(PATHS[i]),
                                       d2=round(float(d2[i]), 5)) for i in near[:6]])
    if self_index is not None:  # a dataset photo: how far is the live embedding from the precomputed one?
        out["max_abs_diff_vs_precomputed"] = float(max(np.abs(EMB[m][self_index] - z[m]).max() for m in MODELS))
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(SITE), **kw)

    def log_message(self, fmt, *args):  # quieter console: only API calls and errors
        if "/api/" in (self.path or "") or (args and str(args[1])[:1] in "45"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/health":
            return self._json(dict(live=True, models=list(MODELS), photos=len(PATHS)))
        if u.path.startswith("/img/"):
            target = (DATA / unquote(u.path[5:])).resolve()
            if DATA.resolve() not in target.parents or not target.is_file():
                return self._json(dict(error="No such photo."), 404)
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/embed":
            return self._json(dict(error="Unknown endpoint."), 404)
        t0 = time.perf_counter()
        q = parse_qs(u.query)
        try:
            if "path" in q:
                rel = q["path"][0]
                if rel not in PATH_INDEX:
                    return self._json(dict(error="That photo is not in the dataset."), 404)
                img, self_index, source = Image.open(DATA / rel), PATH_INDEX[rel], rel
            else:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= MAX_UPLOAD:
                    return self._json(dict(error="Send one image up to 15 MB."), 413)
                img, self_index, source = Image.open(io.BytesIO(self.rfile.read(n))), None, "upload"
            img.load()
        except Exception:
            return self._json(dict(error="That file could not be read as an image. Use a JPEG or PNG."), 400)
        res = describe(embed(img), self_index)
        res.update(source=source, ms=round((time.perf_counter() - t0) * 1000, 1))
        return self._json(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Serving the SD-198 few-shot site at http://{a.host}:{a.port}  (live model loaded, {len(PATHS)} photos indexed)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
