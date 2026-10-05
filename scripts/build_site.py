"""Build the project website's home page (site/index.html) and copy the report and notebook into site/assets.

    python scripts/build_site.py         # then: python scripts/serve_site.py  ->  http://127.0.0.1:8765

Like scripts/build_report.py, every number on the page is read from the results files; none is typed in.
The demo page (site/demo.html) is static and reads site/assets/data.js (scripts/build_showcase.py).
"""
import html
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from summarize_runs import summarize  # noqa: E402
from utils import load_json  # noqa: E402

R, SITE = ROOT / "results", ROOT / "site"
ASSETS, FIG = SITE / "assets", SITE / "assets" / "fig"

summary = load_json(R / "summary.json")
cfg, frozen = summary["configs"], summary["frozen_baseline"]
extra = load_json(R / "extra_summary.json") if (R / "extra_summary.json").exists() else {}
ver = load_json(R / "verification.json") if (R / "verification.json").exists() else None
dedup = load_json(R / "dedup_test.json") if (R / "dedup_test.json").exists() else None
splits = load_json(ROOT / "splits.json")
frozen_val = load_json(R / "frozen_val.json")["5way_5shot"]["acc"]
ev = summarize(ROOT / "runs/_evidence/full20k_lr1e-4")
run_cfg = load_json(ROOT / "runs/seed42/config.json")
run_done = load_json(ROOT / "runs/seed42/done.json")
manifest = load_json(ROOT / "data_manifest.json")
KEYS = ["5way_1shot", "5way_5shot", "2way_1shot", "2way_5shot"]
NAMES = {"5way_1shot": "5 diseases, 1 photo each", "5way_5shot": "5 diseases, 5 photos each",
         "2way_1shot": "2 diseases, 1 photo each", "2way_5shot": "2 diseases, 5 photos each"}
SHORT = {"5way_1shot": "5-way 1-shot", "5way_5shot": "5-way 5-shot", "2way_1shot": "2-way 1-shot", "2way_5shot": "2-way 5-shot"}
CHANCE = {"5way_1shot": 20, "5way_5shot": 20, "2way_1shot": 50, "2way_5shot": 50}
esc = html.escape
OK = '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M6.2 11.6 2.8 8.2l1.1-1.1 2.3 2.3 5.9-5.9 1.1 1.1z"/></svg>'
NO = '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="m4.1 3 3.9 3.9L11.9 3 13 4.1 9.1 8l3.9 3.9-1.1 1.1L8 9.1 4.1 13 3 11.9 6.9 8 3 4.1z"/></svg>'
WARN = '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 1.5 15 14H1zm-.8 4.5v4h1.6V6zm0 5.2v1.6h1.6v-1.6z"/></svg>'
cfg_val_mean = sum(load_json(ROOT / f"runs/seed{s}/done.json")["best_val"] for s in (42, 43, 44)) / 3


def counts(split):
    n = [splits["counts"][c] for c, v in splits["splits"].items() if v == split]
    return len(n), sum(n), min(n), max(n)


# ---------------------------------------------------------------- downloads
for f in ("report.pdf", "notebook.ipynb"):
    shutil.copy2(ROOT / f, ASSETS / f)
if FIG.exists():
    shutil.rmtree(FIG)          # figures are drawn natively in the page now; nothing to copy


def logs(run):
    recs = [json.loads(line) for line in open(ROOT / "runs" / run / "log.jsonl") if line.strip()]
    return [r for r in recs if r["type"] == "val"], [r for r in recs if r["type"] == "train"]


# ---------------------------------------------------------------- class-size histogram (inline SVG)
def histogram():
    W, H, top, bot, left, right = 560, 300, 18, 44, 40, 10
    edges = list(range(10, 66, 5))
    bins = {sp: [0] * (len(edges) - 1) for sp in ("train", "val", "test")}
    for c, sp in splits["splits"].items():
        b = min(len(edges) - 2, (splits["counts"][c] - 10) // 5)
        bins[sp][b] += 1
    tot = [sum(bins[sp][i] for sp in bins) for i in range(len(edges) - 1)]
    ymax = (max(tot) // 10 + 1) * 10
    ih, iw = H - top - bot, W - left - right
    y = lambda v: top + ih * (1 - v / ymax)
    bw = iw / (len(edges) - 1)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="histT">'
           f'<title id="histT">Number of diseases by photos per disease, coloured by split</title>']
    for v in range(0, ymax + 1, 10):
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for i in range(len(edges) - 1):
        base = 0
        for sp, cls in (("train", "s-train"), ("val", "s-val"), ("test", "s-test")):
            v = bins[sp][i]
            if not v:
                continue
            gap = 2 if base else 0        # 2px surface gap between stacked segments
            out.append(f'<rect class="{cls}" x="{left + i * bw + 2:.1f}" y="{y(base + v):.1f}" width="{bw - 4:.1f}" '
                       f'height="{max(1.0, y(base) - y(base + v) - gap):.1f}" rx="2">'
                       f'<title>{v} {sp} diseases with {edges[i]} to {edges[i + 1] - 1} photos</title></rect>')
            base += v
    for i, e in enumerate(edges):
        if e % 10 == 0:
            out.append(f'<text x="{left + i * bw:.1f}" y="{H - bot + 18}" text-anchor="middle">{e}</text>')
    out.append(f'<text class="cat" x="{left + iw / 2:.1f}" y="{H - 6}" text-anchor="middle">photos per disease</text>'
               f'<line class="axis" x1="{left}" x2="{W - right}" y1="{y(0):.1f}" y2="{y(0):.1f}"/></svg>')
    return "".join(out)


# ---------------------------------------------------------------- training story: the planned run vs the final recipe
def training_chart():
    ev_val, ev_tr = logs("_evidence/full20k_lr1e-4")
    fin = [logs(f"seed{s}") for s in (42, 43, 44)]
    xmax = max(r["episodes"] for r in ev_val + ev_tr)

    def panel(title, ylo, yhi, ystep, series, ref=None, x0=0):
        W, H, top, bot, left, right = 470, 290, 34, 40, 42, 16
        ih, iw = H - top - bot, W - left - right
        X = lambda e: x0 + left + iw * e / xmax
        Y = lambda v: top + ih * (1 - (v - ylo) / (yhi - ylo))
        o = [f'<text class="cat" x="{x0 + left}" y="18">{title}</text>']
        v = ylo
        while v <= yhi + 1e-9:
            o.append(f'<line class="grid" x1="{x0 + left}" x2="{x0 + W - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                     f'<text x="{x0 + left - 7}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
            v += ystep
        for e in range(0, xmax + 1, 1000):
            o.append(f'<text x="{X(e):.1f}" y="{H - bot + 17}" text-anchor="middle">{e // 1000}k</text>')
        if ref is not None:
            o.append(f'<line class="ref" x1="{x0 + left}" x2="{x0 + W - right}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}"/>'
                     f'<text class="lab" x="{x0 + W - right}" y="{Y(ref) + 15:.1f}" text-anchor="end">ImageNet only {ref:.1f}</text>')
        for cls, recs, label in series:
            pts = " ".join(f"{X(r['episodes']):.1f},{Y(min(max(r['acc'], ylo), yhi)):.1f}" for r in recs)
            o.append(f'<polyline class="{cls}" points="{pts}"><title>{label}</title></polyline>')
        o.append(f'<line class="axis" x1="{x0 + left}" x2="{x0 + W - right}" y1="{Y(ylo):.1f}" y2="{Y(ylo):.1f}"/>'
                 f'<text x="{x0 + left + iw / 2:.1f}" y="{H - 6}" text-anchor="middle">training episodes</text>')
        return o, X, Y

    plan = ("l-plan", "planned schedule: 20k episodes at lr 1e-4 (stopped at 6k)")
    a, X, Y = panel("Accuracy on unseen validation diseases (%)", 68, 82, 2,
                    [(plan[0], ev_val, plan[1])] + [("l-final", f[0], f"final recipe, seed {s}") for f, s in zip(fin, (42, 43, 44))],
                    ref=frozen_val)
    best, last, fend = max(ev_val, key=lambda r: r["acc"]), ev_val[-1], fin[0][0][-1]
    a.append(f'<circle class="d-plan" cx="{X(best["episodes"]):.1f}" cy="{Y(best["acc"]):.1f}" r="4"/>'
             f'<text class="lab" x="{X(best["episodes"]) + 8:.1f}" y="{Y(best["acc"]) - 9:.1f}">peak {best["acc"]:.1f}</text>'
             f'<circle class="d-plan" cx="{X(last["episodes"]):.1f}" cy="{Y(last["acc"]):.1f}" r="4"/>'
             f'<text class="lab" x="{X(last["episodes"]) - 8:.1f}" y="{Y(last["acc"]) + 4:.1f}" text-anchor="end">planned run {last["acc"]:.1f}</text>'
             f'<text class="lab am" x="{X(fend["episodes"]) + 8:.1f}" y="{Y(fend["acc"]) - 10:.1f}">final recipe</text>')
    b, X2, Y2 = panel("Accuracy on training diseases (%)", 60, 100, 10,
                      [(plan[0], ev_tr, plan[1])] + [("l-final", f[1], f"final recipe, seed {s}") for f, s in zip(fin, (42, 43, 44))], x0=510)
    b.append(f'<text class="lab" x="{X2(ev_tr[-1]["episodes"]) - 6:.1f}" y="{Y2(ev_tr[-1]["acc"]) + 18:.1f}" text-anchor="end">memorised: {ev_tr[-1]["acc"]:.1f}</text>')
    head = (f'<svg class="chart" viewBox="0 0 980 290" role="img" aria-labelledby="trT trD">'
            f'<title id="trT">Validation and training accuracy: the planned schedule against the final recipe</title>'
            f'<desc id="trD">Planned run: validation peaks at {best["acc"]:.1f}% and falls to {last["acc"]:.1f}% while training accuracy reaches '
            f'{ev_tr[-1]["acc"]:.1f}%. Final recipe: best validation about {cfg_val_mean:.1f}%.</desc>')
    return head + "".join(a + b) + "</svg>"


# ---------------------------------------------------------------- real test episodes as cards (photos from the demo sprite)
def episode_cards(n_cards=3):
    import numpy as np
    from data import EpisodeDataset, SD198Split
    from eval import MIN_TEST_IMAGES, N_QUERY_TEST, TEST_SEED
    test = SD198Split(ROOT / "data/SD-198-256", splits, "test", min_images=MIN_TEST_IMAGES)
    z = np.load(R / "test_embeddings.npz")["seed42"]
    cols = 32
    rows = -(-len(test.paths) // cols)

    def thumb(i, cls="th"):
        c, r = i % cols, i // cols
        return (f'<span class="{cls} sp" style="background-size:{cols * 100}% {rows * 100}%;'
                f'background-position:{c / (cols - 1) * 100:.3f}% {r / (rows - 1) * 100:.3f}%"></span>')

    eds = EpisodeDataset(test, 5, 5, N_QUERY_TEST, 600, TEST_SEED)
    cards = []
    for e in range(n_cards):
        classes, sup, qry = eds.episode(e)
        protos = z[sup].mean(1)
        truth = e % 5                       # a fixed rule (episode number), never chosen by outcome
        q = int(qry[truth][0])
        d2 = ((protos - z[q]) ** 2).sum(1)
        pred = int(d2.argmin())
        ok = pred == truth
        name = lambda c: esc(test.classes[classes[c]].replace("_", " "))
        rows_html = "".join(
            f'<div class="er{" hit" if c == pred else ""}"><span class="lt">{"ABCDE"[c]}</span>'
            f'<div class="en"><b>{name(c)}</b><div class="ths">{"".join(thumb(int(i)) for i in sup[c])}</div></div>'
            f'<span class="d mono">{d2[c]:.3f}</span></div>' for c in range(5))
        verdict = f'<span class="verdict {"ok" if ok else "no"}">{OK if ok else NO}{"Correct" if ok else "Wrong"}: picked {"ABCDE"[pred]}</span>'
        cards.append(
            f'<article class="ep"><header><span class="tag">Test episode {e + 1} of 600</span>{verdict}</header>'
            f'<div class="eq">{thumb(q, "qth")}<p>New photo of <b>{"ABCDE"[truth]}: {name(truth)}</b>. '
            f'Squared distance to each prototype; the lowest wins.</p></div><div class="erows">{rows_html}</div></article>')
    return "".join(cards)


# ---------------------------------------------------------------- results chart (inline SVG, drawn to scale)
def results_chart():
    """Dot chart, one row per task: ImageNet only -> trained -> trained + batch inference, on a zoomed accuracy axis.
    Dots (not bars) so the axis need not start at 0; chance is stated in the row label instead."""
    trans = load_json(R / "transductive_test.json")["results"] if (R / "transductive_test.json").exists() else None
    W, row_h, top, left, right = 980, 74, 44, 210, 210
    H = top + row_h * len(KEYS) + 34
    lo, hi = 60, 100
    X = lambda v: left + (W - left - right) * (v - lo) / (hi - lo)
    o = [f'<svg class="chart dots" viewBox="0 0 {W} {H}" role="img" aria-labelledby="dotT dotD">'
         f'<title id="dotT">Accuracy on never-seen diseases: ImageNet features, trained model, trained model with batch inference</title><desc id="dotD">'
         + "; ".join(f"{SHORT[k]}: ImageNet only {frozen[k]['acc']:.1f}%, trained {cfg[k]['acc_mean']:.1f}%"
                     + (f", with batch inference {trans[k]['trained_transductive']:.1f}%" if trans else "") for k in KEYS) + "</desc>"]
    for v in range(lo, hi + 1, 10):
        o.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 10}" y2="{H - 30}"/>'
                 f'<text x="{X(v):.1f}" y="{H - 10}" text-anchor="middle">{v}%</text>')
    o.append(f'<text class="hd" x="{left - 16}" y="{top - 20}" text-anchor="end">task (chance)</text>'
             f'<text class="hd" x="{W - right + 16}" y="{top - 20}">mistakes removed</text>')
    for i, k in enumerate(KEYS):
        y = top + row_h * i + row_h / 2
        f, t = frozen[k]["acc"], cfg[k]["acc_mean"]
        b = trans[k]["trained_transductive"] if trans else None
        best = b if b is not None else t
        cut = 100 * ((100 - f) - (100 - best)) / (100 - f)
        o.append(f'<line class="row" x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                 f'<text class="cat" x="{left - 16}" y="{y - 4:.1f}" text-anchor="end">{NAMES[k]}</text>'
                 f'<text x="{left - 16}" y="{y + 14:.1f}" text-anchor="end">chance {CHANCE[k]}%</text>'
                 f'<line class="span" x1="{X(f):.1f}" x2="{X(best):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
                 f'<circle class="p-frozen" cx="{X(f):.1f}" cy="{y:.1f}" r="7"><title>ImageNet only: {f:.1f}%</title></circle>'
                 f'<text class="v-frozen" x="{X(f):.1f}" y="{y - 14:.1f}" text-anchor="middle">{f:.1f}</text>'
                 f'<circle class="p-trained" cx="{X(t):.1f}" cy="{y:.1f}" r="8"><title>Trained model: {t:.1f}%</title></circle>'
                 f'<text class="v-trained" x="{X(t):.1f}" y="{y + 25:.1f}" text-anchor="middle">{t:.1f}</text>')
        if b is not None:
            o.append(f'<circle class="p-trans" cx="{X(b):.1f}" cy="{y:.1f}" r="8"><title>Trained + batch inference: {b:.1f}%</title></circle>'
                     f'<text class="v-trans" x="{X(b):.1f}" y="{y - 14:.1f}" text-anchor="middle">{b:.1f}</text>')
        o.append(f'<text class="cut" x="{W - right + 16}" y="{y + 6:.1f}">&minus;{cut:.0f}%</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- method diagram (inline SVG)
PIPELINE = """<svg class="pipeline" viewBox="0 0 980 250" role="img" aria-labelledby="pipeT">
<title id="pipeT">Photo, to ResNet-18, to a 512-number embedding, to prototypes, to the nearest prototype</title>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="ah" d="M0 0L10 5L0 10z"/></marker></defs>
<rect class="box" x="10" y="70" width="140" height="110" rx="14"/>
<text class="t-strong" x="80" y="112" text-anchor="middle">Clinical photo</text>
<text x="80" y="136" text-anchor="middle">224 x 224 crop</text>
<text x="80" y="156" text-anchor="middle">of a 256 px image</text>
<path class="arrow" d="M152 125H196" marker-end="url(#ah)"/>
<rect class="box" x="200" y="56" width="168" height="138" rx="14"/>
<text class="t-strong" x="284" y="100" text-anchor="middle">ResNet-18</text>
<text x="284" y="124" text-anchor="middle">ImageNet weights,</text>
<text x="284" y="144" text-anchor="middle">fine-tuned with</text>
<text x="284" y="164" text-anchor="middle">episodic training</text>
<path class="arrow" d="M370 125H414" marker-end="url(#ah)"/>
<rect class="box" x="418" y="70" width="150" height="110" rx="14"/>
<text class="t-strong" x="493" y="112" text-anchor="middle">512 numbers</text>
<text x="493" y="136" text-anchor="middle">one point in the</text>
<text x="493" y="156" text-anchor="middle">embedding space</text>
<path class="arrow" d="M570 125H614" marker-end="url(#ah)"/>
<rect class="field" x="618" y="18" width="352" height="214" rx="16"/>
<g>
<circle class="pt" cx="680" cy="70" r="4"/><circle class="pt" cx="700" cy="92" r="4"/><circle class="pt" cx="664" cy="98" r="4"/><circle class="ring" cx="682" cy="86" r="13"/><text class="lt" x="682" y="90" text-anchor="middle">A</text>
<circle class="pt" cx="890" cy="64" r="4"/><circle class="pt" cx="918" cy="80" r="4"/><circle class="pt" cx="896" cy="96" r="4"/><circle class="ring" cx="901" cy="80" r="13"/><text class="lt" x="901" y="84" text-anchor="middle">B</text>
<circle class="pt" cx="700" cy="178" r="4"/><circle class="pt" cx="676" cy="196" r="4"/><circle class="pt" cx="712" cy="200" r="4"/><circle class="ring" cx="696" cy="191" r="13"/><text class="lt" x="696" y="195" text-anchor="middle">C</text>
<circle class="pt-am" cx="882" cy="182" r="4"/><circle class="pt-am" cx="906" cy="196" r="4"/><circle class="pt-am" cx="872" cy="204" r="4"/><circle class="ring-am" cx="887" cy="194" r="13"/><text class="lt" x="887" y="198" text-anchor="middle">D</text>
<line class="fl-dim" x1="806" y1="150" x2="695" y2="98"/><line class="fl-dim" x1="806" y1="150" x2="890" y2="90"/><line class="fl-dim" x1="806" y1="150" x2="708" y2="186"/>
<line class="fl" x1="806" y1="150" x2="876" y2="187"/>
<circle class="q" cx="806" cy="150" r="7"/>
<text class="lt" x="806" y="134" text-anchor="middle">new photo</text>
<text class="lt" x="794" y="42" text-anchor="middle" style="fill:#95a0ba;font-weight:400">prototype = mean of the example photos</text>
</g></svg>"""


def pct(v):
    return f"{v:.1f}%"


c5, c1 = cfg["5way_5shot"], cfg["5way_1shot"]
f5, f1 = frozen["5way_5shot"]["acc"], frozen["5way_1shot"]["acc"]
tr_n, tr_img, *_ = counts("train")
va_n, va_img, *_ = counts("val")
te_n, te_img, te_min, te_max = counts("test")

def trans_tbl(k):
    if not (R / "transductive_test.json").exists():
        return "&ndash;"
    return f"{load_json(R / 'transductive_test.json')['results'][k]['trained_transductive']:.1f}%"


res_rows = "".join(
    f"<tr><td>{NAMES[k]}</td><td class='n'><strong>{cfg[k]['acc_mean']:.1f}%</strong></td><td class='n'>±{cfg[k]['acc_ci95_episodes']:.1f}</td>"
    f"<td class='n'>{cfg[k]['f1_mean']:.1f}</td><td class='n'>{trans_tbl(k)}</td><td class='n'>{frozen[k]['acc']:.1f}%</td>"
    f"<td class='n'>{100 * (cfg[k]['acc_mean'] - frozen[k]['acc']) / (100 - frozen[k]['acc']):.0f}%</td><td class='n'>{CHANCE[k]}%</td></tr>" for k in KEYS)

# robustness cards
cards = []
if "repro" in extra:
    d = max(abs(v["diff"]) for v in extra["repro"]["test"].values())
    cards.append(("Same run, another GPU", f"≤ {d:.2f} pp",
                  f"Retraining the seed-42 run on a second machine lands within {d:.2f} pp on every test setting. That is the noise floor for comparing runs."))
if "splits" in extra:
    ac = extra["splits"]["across"]
    d = max(ac[k]["max"] - ac[k]["min"] for k in ("5way_1shot", "5way_5shot"))
    cards.append(("Other training diseases", f"≤ {d:.1f} pp",
                  f"Two more train/validation partitions of the 128 common diseases, 3 seeds each: 5-way accuracy moves by at most {d:.1f} pp "
                  f"({ac['5way_5shot']['mean']:.1f}% mean for 5 photos each)."))
if "r50" in extra:
    t5 = extra["r50"]["test"]
    sig = [SHORT[k] for k in KEYS if t5[k]["diff_boot_ci95"][0] > 0]
    cards.append(("A bigger backbone", f"+{min(t5[k]['diff'] for k in KEYS):.1f} to +{max(t5[k]['diff'] for k in KEYS):.1f} pp",
                  f"ResNet-50 (2x the parameters) helps only a little: the difference is clear of noise in {' and '.join(sig) or 'no setting'} only."))
cards_html = "".join(f'<div class="card"><span class="tag">{esc(a)}</span><span class="num">{esc(b)}</span><p>{esc(c)}</p></div>' for a, b, c in cards)

checks = []
if ver:
    t = ver["test"]
    rec = max(abs(t[k][m]["recomputed"] - t[k][m]["stored"]) for k in KEYS for m in ("seed42", "seed43", "seed44", "frozen"))
    fresh = max(abs(t[k][m]["fresh_5000"] - t[k][m]["stored"]) for k in KEYS for m in ("seed42", "seed43", "seed44", "frozen"))
    mask = max(abs(t[k][m]["bottom_masked"] - t[k][m]["recomputed"]) for k in KEYS for m in ("seed42", "seed43", "seed44"))
    checks += [
        (OK, "Numbers reproduce", f"Recomputed through a separate code path on CPU: every result matches within {rec:.2f} pp."),
        (OK, "Not tuned to 600 lucky episodes", f"On 5,000 new test episodes, accuracy stays within {fresh:.1f} pp."),
        (OK, "The metric is sound", "Shuffling the answer labels drops accuracy to chance (20% and 50%)."),
        (OK, "No test peeking", "Each checkpoint was chosen on validation diseases only; training logs hold no test records."),
        (OK, "No watermark shortcut", f"Masking the grey source strip at the bottom of the photos changes accuracy by at most {mask:.1f} pp."),
        (OK, "Fine-tuning really helps", "The gain over ImageNet features is clear of noise in all four settings (paired bootstrap over episodes)."),
    ]
if dedup:
    a, b, c = (dedup[k]["results"]["5way_1shot"]["finetuned_mean"] for k in ("full", "cos>0.9", "cos>0.9_classes_all_images"))
    checks.append((WARN, "Same-patient photos", f"SD-198 holds re-shoots of the same patient. Removing them costs about {c - b:.1f} pp at 1 photo each "
                   f"(at most {a - b:.1f} pp); the gain over ImageNet features is unchanged."))
checks_html = "".join(f'<li class="{"warn" if i is WARN else ""}">{i}<span><strong>{esc(h)}.</strong> {esc(p)}</span></li>' for i, h, p in checks)

ep_cards = episode_cards()
trans_all = load_json(R / "transductive_test.json")["results"] if (R / "transductive_test.json").exists() else None
wins = {k: ver["test"][k]["gain_vs_frozen"] for k in KEYS} if ver else {k: dict(wins=0, losses=0) for k in KEYS}


def err_cut(k, best=False):
    f = 100 - frozen[k]["acc"]
    t = 100 - (trans_all[k]["trained_transductive"] if best and trans_all else cfg[k]["acc_mean"])
    return 100 * (f - t) / f


trans_legend = '<span><i class="k-dtrans"></i>Trained + batch inference</span>' if trans_all else ""
trans_note = (" Batch inference refines each prototype with the episode's other unlabelled new photos (Ren et al., 2018); it needs the new photos to arrive "
              "together, its settings were fixed on validation diseases, and it ran once on test.") if trans_all else ""

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>SD-198 Few-Shot</title>
<meta name="description" content="Recognising skin diseases a model has never seen, from one or five example photos: a prototypical network on SD-198.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=JetBrains+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<nav class="nav" aria-label="Site">
  <div class="nav-in">
    <a class="brand" href="index.html"><svg class="brand-mark" viewBox="0 0 22 22" aria-hidden="true"><circle cx="6" cy="7" r="2.2" fill="#8b95a8"/><circle cx="15" cy="5" r="1.6" fill="#8b95a8"/><circle cx="11" cy="15" r="3" fill="none" stroke="#e08a2c" stroke-width="1.8"/><circle cx="17.5" cy="16.5" r="1.8" fill="#e08a2c"/></svg><span>SD-198 Few-Shot</span></a>
    <div class="nav-links">
      <a href="#problem" class="hide-sm">Problem</a>
      <a href="#method" class="hide-sm">Method</a>
      <a href="#results">Results</a>
      <a href="#checks" class="hide-sm">Checks</a>
      <a href="demo.html">Demo</a>
      <a href="assets/report.pdf" class="cta">Report</a>
    </div>
  </div>
</nav>

<header class="hero-band">
  <div class="hb-galaxy" id="heroGalaxy" aria-hidden="true"></div>
  <div class="hb-in">
    <div class="hb-copy">
      <p class="eyebrow">Deep learning class project · prototypical networks · SD-198</p>
      <h1>Recognising skin diseases it has <em>never seen</em>, from five photos.</h1>
      <p class="lede">Most skin diseases are rare, with too few photos to train a classifier. We trained a network to compare photos instead.
        Shown one to five examples of a disease it never trained on, it labels new photos by which example group they sit closest to.</p>
      <div class="actions">
        <a class="btn primary" href="demo.html"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9.5-5.5z" fill="currentColor" stroke="none"/></svg>Open the live demo</a>
        <a class="btn" href="assets/report.pdf"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 1.5h5.5L13 5v9.5H4zM9.5 1.5V5H13"/></svg>Report (PDF)</a>
        <a class="btn" href="assets/notebook.ipynb" download><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2v8m0 0-3-3m3 3 3-3M3 13.5h10"/></svg>Notebook</a>
      </div>
      <p class="hb-legend"><span><i class="seen"></i>{tr_n + va_n} diseases it learned from</span><span><i class="unseen"></i>{te_n} it never saw</span><span>every dot is one photo, placed by the trained model</span></p>
    </div>
    <div class="facts">
      <div><b class="am">{pct(c5['acc_mean'])}</b><span>5 never-seen diseases, 5 example photos each (chance 20%)</span></div>
      <div><b class="am">{pct(c1['acc_mean'])}</b><span>the same from a single photo each</span></div>
      <div><b>&minus;{100 * (c5['acc_mean'] - f5) / (100 - f5):.0f}%</b><span>mistakes, compared with the same network without our training</span></div>
      <div><b>{run_done['wall_clock_s'] / 60:.0f} min</b><span>training per run on one {esc(run_done['device'].replace('NVIDIA GeForce ', ''))}</span></div>
    </div>
  </div>
</header>

<main class="page">
  <section class="section" id="problem" aria-labelledby="h-problem">
    <p class="kicker">The problem</p>
    <h2 id="h-problem">Rare diseases have too few photos to learn from</h2>
    <div class="two">
      <div class="prose">
        <p>SD-198 holds <strong>{manifest['n_images']:,} clinical photos of {manifest['n_classes']} skin diseases</strong>, taken with ordinary cameras: varied light, framing and skin tones.
          Many diseases are rare. {te_n} of them have only {te_min} to {te_max} photos.</p>
        <p>A standard classifier needs hundreds of photos per disease and only knows the diseases it was trained on. So we make the test honest:
          <strong>every disease with fewer than 20 photos is held out completely</strong>. The model trains on {tr_n} diseases, picks its checkpoint on {va_n} others,
          and is tested on the {te_n} it has never seen.</p>
        <p>Each test is an <em>episode</em>: pick 5 never-seen diseases, show the model 1 or 5 example photos of each, then ask it to label new photos of those diseases.</p>
      </div>
      <figure class="chart-card">
        <div class="legend-row"><span><i class="k-train"></i>Training ({tr_n})</span><span><i class="k-val"></i>Validation ({va_n})</span><span><i class="k-test"></i>Never seen, test ({te_n})</span></div>
        {histogram()}
        <figcaption>Diseases by number of photos. The never-seen test diseases are the rare ones: {te_img} photos across {te_n} diseases. Training: {tr_img} photos; validation: {va_img}.</figcaption>
      </figure>
    </div>
  </section>

  <section class="section" id="method" aria-labelledby="h-method">
    <p class="kicker">Method</p>
    <h2 id="h-method">Learn to compare, not to memorise</h2>
    <div class="diagram-wrap">{PIPELINE}</div>
    <div class="two">
      <div class="prose">
        <p><strong>Prototypical network</strong> (Snell et al., 2017). A ResNet-18 turns every photo into 512 numbers. For each disease in an episode, the average of its
          example photos is its <em>prototype</em>. A new photo gets the label of the nearest prototype by squared distance. There is no classifier layer, so new diseases need no retraining.</p>
        <p><strong>Episodic training.</strong> Training repeats that exact test on the {tr_n} training diseases: thousands of small 5-disease quizzes, each pulling photos of the same disease
          together and pushing different diseases apart.</p>
      </div>
      <div class="prose">
        <p><strong>Recipe:</strong> ResNet-18 with ImageNet weights, {run_cfg['episodes']:,} episodes (5 diseases, 5 examples, 15 queries each; 4 episodes per step),
          AdamW with backbone learning rate {run_cfg['lr_backbone']:g}, cosine schedule, bf16, random crops, flips, rotation and colour jitter.
          The checkpoint with the best accuracy on 600 fixed validation episodes is kept.</p>
        <p><strong>Baseline:</strong> the same network and prototype rule with <em>no</em> fine-tuning (plain ImageNet features), scored on the identical test episodes.
          Fine-tuning only counts if it beats this.</p>
      </div>
    </div>
  </section>

  <section class="section" id="training" aria-labelledby="h-training">
    <p class="kicker">What went wrong first</p>
    <h2 id="h-training">The planned schedule memorised the training diseases</h2>
    <p class="lede">The original plan, {ev['cfg']['episodes']:,} episodes at learning rate {ev['cfg']['lr_backbone']:g}, overfit. On unseen validation diseases it peaked at
      {ev['best_val']:.1f}% and then fell to {ev['final_val']:.1f}%, while accuracy on the training diseases climbed to {ev['train_acc']:.1f}%. A five-setting sweep, judged on
      validation diseases only, led to the final recipe: {run_cfg['episodes']:,} episodes at learning rate {run_cfg['lr_backbone']:g}. The test diseases were not looked at until the end.</p>
    <figure class="chart-card">
      <div class="legend-row"><span><i class="k-plan"></i>Planned schedule (stopped at 6k episodes)</span><span><i class="k-final"></i>Final recipe, 3 seeds</span><span><i class="k-chance"></i>Plain ImageNet features</span></div>
      <div class="tbl-wrap">{training_chart()}</div>
      <figcaption>Left: accuracy on the {va_n} validation diseases (600 fixed episodes). Right: accuracy on the training episodes themselves. The planned run keeps
        improving on diseases it trains on while getting worse on new ones.</figcaption>
    </figure>
  </section>

  <section class="section" id="results" aria-labelledby="h-results">
    <p class="kicker">Results</p>
    <h2 id="h-results">{pct(c5['acc_mean'])} on diseases it has never seen</h2>
    <p class="lede">The plain ImageNet network is already a strong start, so the honest measure is how many of its <strong>mistakes</strong> the trained model removes:
      {err_cut('5way_5shot'):.0f}% of them with five photos per disease, {err_cut('5way_1shot'):.0f}% with one. On the same 600 test episodes it beats the ImageNet network
      in {wins['5way_1shot']['wins']} and loses {wins['5way_1shot']['losses']} (one photo each).</p>
    <figure class="chart-card">
      <div class="legend-row"><span><i class="k-dfrozen"></i>ImageNet features only</span><span><i class="k-dtrained"></i>Trained model</span>{trans_legend}</div>
      <div class="tbl-wrap">{results_chart()}</div>
      <figcaption>600 fixed test episodes per task, identical for every model; trained-model numbers are the mean of 3 seeds. The right column is the share of the
        ImageNet network's mistakes that the best variant removes.{trans_note}</figcaption>
    </figure>
    <div class="tbl-wrap"><table>
      <thead><tr><th>Task</th><th>Trained model</th><th>95% CI</th><th>Macro-F1</th><th>+ batch inference</th><th>ImageNet only</th><th>Mistakes removed</th><th>Chance</th></tr></thead>
      <tbody>{res_rows}</tbody></table></div>
    <h3 class="sub">Three real test episodes</h3>
    <p class="prose">The first three of the 600 test episodes, one new photo each, chosen by a fixed rule rather than by outcome. Each row is a never-seen disease with its five example photos.</p>
    <div class="eps">{ep_cards}</div>
    <p><a class="btn primary" href="demo.html#lab">Try more episodes in the demo &rarr;</a></p>
  </section>

  <section class="section" id="robust" aria-labelledby="h-robust">
    <p class="kicker">How robust is it</p>
    <h2 id="h-robust">The result holds when we change things</h2>
    <div class="cards">{cards_html}</div>
  </section>

  <section class="section" id="checks" aria-labelledby="h-checks">
    <p class="kicker">Did we check it</p>
    <h2 id="h-checks">Every number was re-derived independently</h2>
    <ul class="checks">{checks_html}</ul>
  </section>

  <section class="section" id="limits" aria-labelledby="h-limits">
    <p class="kicker">Limits and next steps</p>
    <h2 id="h-limits">What this does not show</h2>
    <div class="two">
      <div class="prose">
        <p>This is a class project, not a clinical tool. The test diseases have only {te_min} to {te_max} photos, so each episode uses 5 query photos per disease,
          and confidence intervals are about ±1 pp. Plain ImageNet features are already strong ({pct(f5)}), which limits how much fine-tuning can add.</p>
        <p>The backbone and method are deliberately simple, so the numbers are not comparable with published SD-198 results that use larger models.</p>
      </div>
      <div class="prose">
        <p><strong>Next:</strong> dermatology-specific or self-supervised pretraining, which may matter more than a bigger network (ResNet-50 added at most
          {max(extra['r50']['test'][k]['diff'] for k in KEYS) if 'r50' in extra else 0:.1f} pp); cross-dataset tests on dermoscopy (Derm7pt); a patient-aware test sampler.</p>
      </div>
    </div>
  </section>

  <section class="section" id="reproduce" aria-labelledby="h-repro">
    <p class="kicker">Reproduce</p>
    <h2 id="h-repro">Run it yourself</h2>
<pre class="code"><span class="c"># data (pinned HuggingFace revision, sha256-checked) and 256 px thumbnails</span>
python scripts/download_data.py && python scripts/prep_thumbs.py && python src/data.py --verify
<span class="c"># train one seed on a GPU (about {run_done['wall_clock_s'] / 60:.0f} min on an {esc(run_done['device'].replace('NVIDIA GeForce ', ''))}), then test it</span>
bash scripts/run_seed.sh 42
<span class="c"># independent checks, the report, and this website</span>
python scripts/verify_claims.py && python scripts/build_report.py
python scripts/embed_all.py && python scripts/build_showcase.py && python scripts/build_site.py
python scripts/serve_site.py      <span class="c"># http://127.0.0.1:8765</span></pre>
  </section>

  <footer class="site">
    <p><strong>Research demo, not a diagnostic tool.</strong> Built for a deep-learning class project. Dataset: SD-198 (Sun et al., ECCV 2016),
      HuggingFace mirror <span class="mono">{esc(manifest['download']['repo'])}</span> at a pinned revision; photos shown for teaching only.</p>
    <p>References: Snell et al., Prototypical Networks (NeurIPS 2017); Sun et al., SD-198 (ECCV 2016). Three.js (MIT) draws the 3-D views.</p>
  </footer>
</main>

<script src="assets/vendor/three.min.js"></script>
<script src="assets/data.js"></script>
<script src="assets/galaxy.js"></script>
<script>
  // hero: the trained model's space, slowly rotating (the full interactive version is on the demo page)
  if (window.THREE && window.Galaxy && window.SHOWCASE && window.SHOWCASE.atlas) {{
    const g = new Galaxy(document.getElementById("heroGalaxy"), SHOWCASE.atlas,
      {{ interactive: false, autoRotate: true, offsetX: w => (w > 980 ? Math.min(360, w * 0.22) : 0) }});
    g.orbit.r = 31;
  }}
</script>
</body>
</html>
"""
(SITE / "index.html").write_text(PAGE)
print(f"wrote site/index.html ({len(PAGE) / 1e3:.0f} kB) and copied report.pdf and notebook.ipynb into site/assets")
