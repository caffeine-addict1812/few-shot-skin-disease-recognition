"""Figures: training curves, t-SNE of novel-class embeddings, qualitative episodes, distance bars.

Colors follow the dataviz reference palette (categorical slots in fixed order, light surface).
Scatter plots use color + marker shape (composite encoding): past 3 hues an all-pairs scatter
cannot rely on color alone.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b2"
GRID, AXIS = "#e9e8e4", "#d3d2cc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "regular", "axes.titlesize": 12,
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "font.size": 10, "figure.dpi": 110,
}


def _pretty(name):
    return name.replace("_", " ")


def _save(fig, out_path):
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------ training curves
def read_log(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in recs if r["type"] == "train"], [r for r in recs if r["type"] == "val"]


def _smooth(y, w):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def plot_training_curves(run_dirs, out_path=None, smooth=5):
    """Two panels side by side (never a dual axis): train loss, and val accuracy per seed."""
    with plt.rc_context(STYLE):
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10.5, 3.8))
        for i, d in enumerate(run_dirs):
            d = Path(d)
            tr, va = read_log(d / "log.jsonl")
            label = f"seed {json.load(open(d / 'config.json'))['seed']}"
            c = SERIES[i]
            ax_l.plot([r["step"] for r in tr], _smooth([r["loss"] for r in tr], smooth),
                      color=c, lw=1.8, label=label)
            ax_a.plot([r["step"] for r in va], [r["acc"] for r in va], color=c, lw=1.8,
                      marker="o", ms=5, mec=SURFACE, mew=1.0, label=label)
        ax_l.set(xlabel="optimizer step", ylabel="episode cross-entropy", title="Training loss")
        ax_a.set(xlabel="optimizer step", ylabel="accuracy (%)",
                 title="Validation accuracy (5-way 5-shot, unseen classes)")
        ax_l.set_ylim(bottom=0)
        handles, labels = ax_l.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout()
        return _save(fig, out_path)


# ------------------------------------------------------------------ embeddings / t-SNE
@torch.no_grad()
def extract_embeddings(model, dataset, class_ids, device, batch=64):
    """Embeddings [n, D] and class ids [n] for every image of `class_ids` (indices into dataset.classes)."""
    model.eval()
    idx = np.concatenate([dataset.class_indices[c] for c in class_ids])
    labels = np.concatenate([[c] * len(dataset.class_indices[c]) for c in class_ids])
    out = []
    for s in range(0, len(idx), batch):
        x = torch.stack([dataset[int(j)][0] for j in idx[s:s + batch]]).to(device)
        out.append(model.embed(x).cpu())
    return torch.cat(out).numpy(), labels


def pick_classes(dataset, n=8):
    """The n classes with the most images (ties by index): the best-populated novel classes."""
    order = sorted(range(len(dataset.classes)), key=lambda c: (-len(dataset.class_indices[c]), c))
    return sorted(order[:n])


def plot_tsne(emb, labels, class_names, out_path=None, seed=0, title="t-SNE of novel-class embeddings"):
    from sklearn.manifold import TSNE

    uniq = sorted(set(labels.tolist()))
    assert len(uniq) <= len(SERIES), f"at most {len(SERIES)} classes (categorical palette cap)"
    perp = max(5, min(30, (len(emb) - 1) // 3))
    xy = TSNE(2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(emb)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        for slot, c in enumerate(uniq):  # slot follows the class, not its rank in any filter
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], s=46, color=SERIES[slot], marker=MARKERS[slot],
                       edgecolors=SURFACE, linewidths=0.8, label=_pretty(class_names[c]))
        ax.set(xticks=[], yticks=[], title=title)
        ax.grid(False)
        ax.set_xlabel("unseen classes only; the model never trained on them", color=INK2)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=1.1, fontsize=9)
        fig.tight_layout()
        return _save(fig, out_path)


# ------------------------------------------------------------------ episodes
def _thumb(dataset, i, size=128):
    img = dataset.load_raw(int(i))
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size))


@torch.no_grad()
def episode_distances(model, dataset, support, query, device):
    """Squared distances [N*Q, N] from every query to the N prototypes for one episode."""
    n_way, k = support.shape
    q = query.shape[1]
    order = np.concatenate([support.ravel(), query.ravel()])
    x = torch.stack([dataset[int(j)][0] for j in order]).unsqueeze(0).to(device)
    logits = model.eval()(x, n_way, k, q)[0]
    return (-logits / model.scale).cpu().numpy()


def plot_episode(model, dataset, episode, device, out_path=None, query_slot=0):
    """Support grid (rows = classes) plus one query per class with predicted / true class and margin.

    margin = squared distance to the 2nd-nearest prototype minus to the nearest (bigger = more confident).
    """
    classes, support, query = episode
    n_way, k = support.shape
    d2 = episode_distances(model, dataset, support, query, device)
    q = query.shape[1]
    names = [dataset.classes[c] for c in classes]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(1.7 * max(k, n_way) + 2.4, 1.7 * n_way + 3.4))
        gs = GridSpec(n_way + 2, max(k, n_way), figure=fig, height_ratios=[1] * n_way + [0.55, 1.4],
                      hspace=0.08, wspace=0.05, left=0.24, right=0.99, top=0.94, bottom=0.01)
        for r in range(n_way):
            for j in range(k):
                ax = fig.add_subplot(gs[r, j])
                ax.imshow(_thumb(dataset, support[r, j]))
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                if j == 0:
                    ax.set_ylabel(f"{r + 1}. {_pretty(names[r])}", rotation=0, ha="right", va="center", fontsize=8.5)
        for r in range(n_way):
            ax = fig.add_subplot(gs[n_way + 1, r])
            ax.imshow(_thumb(dataset, query[r, query_slot]))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            dist = d2[r * q + query_slot]
            order = np.argsort(dist)
            pred, margin = order[0], dist[order[1]] - dist[order[0]]
            ok = pred == r
            ax.set_title(f"true {r + 1}\npred {pred + 1}  {'correct' if ok else 'WRONG'}\nmargin {margin:.2f}",
                         fontsize=8.5, loc="center", color=INK if ok else INK, pad=4,
                         fontweight="regular" if ok else "bold")
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(AXIS if ok else INK)
                s.set_linewidth(1 if ok else 2.5)
        fig.suptitle(f"{n_way}-way {k}-shot episode: support (top), one query per class (bottom)",
                     x=0.01, y=0.985, ha="left", fontsize=12)
        return _save(fig, out_path)


def _draw_distance_bars(ax, d2, class_names, true_idx=None):
    """Horizontal bars of squared distance (lower = closer); the predicted class is the accent."""
    d2 = np.asarray(d2, float)
    pred = int(d2.argmin())
    ys = np.argsort(np.argsort(d2))  # rank 0 (nearest) plotted at the top (axis is inverted below)
    ax.barh(ys, d2, height=0.55, color=[SERIES[0] if i == pred else MUTED for i in range(len(d2))])
    for yi, v in zip(ys, d2):
        ax.text(v + d2.max() * 0.015, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([_pretty(n) + ("  (true)" if i == true_idx else "") + ("  <- predicted" if i == pred else "")
                        for i, n in enumerate(class_names)], fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="squared distance to prototype (lower = closer)", xlim=(0, d2.max() * 1.15))
    ax.grid(axis="y", visible=False)


def plot_distance_bars(d2, class_names, true_idx=None, out_path=None, title="Distance to each class prototype"):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(d2) + 1.4))
        _draw_distance_bars(ax, d2, class_names, true_idx)
        ax.set_title(title)
        fig.tight_layout()
        return _save(fig, out_path)


def plot_demo(dataset, episode, query_img, d2, true_idx=None, out_path=None):
    """One few-shot decision: the support set (rows = classes) | the query photo | distances to prototypes.

    query_img: PIL image of the query; d2: squared distances [N] from that query to the N prototypes.
    """
    classes, support, _ = episode
    n_way, k = support.shape
    names = [dataset.classes[c] for c in classes]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16, 1.5 * n_way + 0.8))
        outer = fig.add_gridspec(1, 3, width_ratios=[k * 1.0, 1.7, 4.2], wspace=0.85)
        sup = outer[0].subgridspec(n_way, k, hspace=0.06, wspace=0.04)
        for r in range(n_way):
            for j in range(k):
                ax = fig.add_subplot(sup[r, j])
                ax.imshow(_thumb(dataset, support[r, j], 112))
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    ax.set_ylabel(f"{r + 1}. {_pretty(names[r])}", rotation=0, ha="right", va="center", fontsize=8.5)
                if r == 0 and j == k // 2:
                    ax.set_title(f"support: {k} photos per class", fontsize=10, color=INK2)
        axq = fig.add_subplot(outer[1])
        w, h = query_img.size
        s = min(w, h)
        axq.imshow(query_img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((224, 224)))
        axq.set_xticks([]), axq.set_yticks([]), axq.grid(False)
        axq.set_title("query photo", fontsize=10, color=INK2)
        axb = fig.add_subplot(outer[2])
        _draw_distance_bars(axb, d2, names, true_idx)
        axb.set_title("nearest prototype wins", fontsize=10, color=INK2, loc="left")
        return _save(fig, out_path)


def plot_split_histogram(counts, splits, out_path=None):
    """Images-per-class histogram, stacked by split: the rare classes are the test classes."""
    names = ("train", "val", "test")
    data = [[counts[c] for c, v in splits.items() if v == n] for n in names]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.hist(data, bins=list(range(10, 66, 5)), stacked=True, color=SERIES[:3], edgecolor=SURFACE,
                linewidth=1.5, label=[f"{n} ({len(d)} classes)" for n, d in zip(names, data)])
        ax.set(xlabel="images per class", ylabel="number of classes",
               title="SD-198 class sizes: novel (test) classes are the rare ones")
        ax.legend(loc="upper center")
        fig.tight_layout()
        return _save(fig, out_path)
