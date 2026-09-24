"""
Training monitor for ``mppp.mask.train``: debug PNGs + CSV, optional live window.

Written to ``<debug_dir>/``:

* ``log.csv``            one row per print interval (interval and running means)
* ``val/ep{E}_it{I}.png``    the SAME fixed held-out frames every time, so
                         panels are comparable across the run (the useful one)
* ``batch/ep{E}_it{I}.png``  first image of the current training batch
                         (panels: image | truth | P(terrain) | prediction − truth)
* ``val_preview.csv``    held-out preview IoU per frame
* ``curves.png``         loss, IoU (train vs held-out), lr, pre-BN ASPP peak

A folder left by an earlier run of the same checkpoint name is moved aside to
``<name>_debug_prev_<time>`` first, so logs never mix runs.  The live window
(``live=True`` or ``monitor.py watch <dir>``) is a separate process that
re-reads these PNGs; it needs no GUI support in OpenCV.

Process isolation (v0p3 fix): the training process never imports matplotlib.
In v0p3's first form, the first in-process matplotlib figure after torch/CUDA
had loaded killed the Jupyter kernel on Windows (native crash, no traceback;
consistent with the duplicate Intel OpenMP runtime, "OMP: Error #15").  Panels
are therefore drawn with OpenCV only — already loaded by the data loader, same
interpolation modes — and ``curves.png`` is rendered by a separate Python
process running this file, so a plotting crash cannot stop training.  The
same renderer can be run by hand on any debug folder::

    python -m mppp.mask.monitor curves checkpoints/<name>_debug

This file must stay importable as a plain script: numpy / csv / stdlib only at
module level, no package-relative imports.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# BGR for OpenCV; hex for matplotlib.  Reference dataviz slots 1-2 + neutral ink.
C_TRAIN, C_VAL = "#2a78d6", "#eb6834"
INK_HEX, MUTED_HEX, GRID_HEX = "#0b0b0b", "#52514e", "#e4e3df"
INK, MUTED, FRAME = (11, 11, 11), (78, 81, 82), (223, 227, 228)
MISSED_BGR, FALSE_BGR = (40, 40, 220), (214, 120, 42)            # red = missed terrain, blue = false terrain


# ---------------------------------------------------------------- statistics
def panel_stats(gt: np.ndarray, prob: np.ndarray, threshold: float) -> Tuple[float, float, float, np.ndarray]:
    """IoU, % missed terrain, % false terrain and the signed difference map, at full resolution."""
    g = gt > 0
    p = prob > threshold
    union = np.logical_or(p, g).sum()
    # v0p12: no terrain in truth and prediction is perfect agreement (IoU 1), not 0
    iou = float(np.logical_and(p, g).sum() / union) if union else 1.0
    diff = p.astype(np.int8) - g.astype(np.int8)
    return iou, 100.0 * float((diff < 0).mean()), 100.0 * float((diff > 0).mean()), diff


# -------------------------------------------------------- OpenCV panel figure
def _tile_h(img: np.ndarray, h: int, nearest: bool = False) -> np.ndarray:
    import cv2
    H, W = img.shape[:2]
    w = max(1, int(round(W * h / H)))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def _text(img, s, org, scale=0.45, color=INK, thick=1):
    import cv2
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_rows(rows: Sequence[tuple], title: str, threshold: float, tile_h: int = 300) -> Tuple[np.ndarray, List[float]]:
    """
    rows: (image_rgb uint8, gt, prob float, label).  Returns a BGR mosaic and
    per-row IoU.  Columns: image | truth | P(terrain) | prediction − truth.
    """
    import cv2
    gap, head, row_head = 8, 34, 20
    tiles, ious = [], []
    for img, gt, prob, label in rows:
        iou, missed, false, diff = panel_stats(gt, prob, threshold)
        ious.append(iou)
        a = _tile_h(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR), tile_h)
        b = _tile_h(((gt > 0) * 255).astype(np.uint8), tile_h, nearest=True)
        c = _tile_h((np.clip(prob, 0, 1) * 255).astype(np.uint8), tile_h)
        d = _tile_h(diff, tile_h, nearest=True)
        dc = np.full(d.shape + (3,), 255, np.uint8)
        dc[d < 0] = MISSED_BGR
        dc[d > 0] = FALSE_BGR
        b, c = (cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in (b, c))
        for t in (a, b, c, dc):
            cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), FRAME, 1)
        cv2.rectangle(dc, (4, 4), (170, 62), (255, 255, 255), -1)
        cv2.rectangle(dc, (4, 4), (170, 62), FRAME, 1)
        for k, s in enumerate((f"IoU {iou:.3f}", f"missed {missed:.2f}%", f"false {false:.2f}%")):
            _text(dc, s, (10, 22 + 17 * k), 0.45)
        sep = np.full((tile_h, gap, 3), 255, np.uint8)
        line = np.hstack([a, sep, b, sep, c, sep, dc])
        lab = np.full((row_head, line.shape[1], 3), 255, np.uint8)
        _text(lab, str(label)[:90], (2, 15), 0.42, MUTED)
        tiles.append(np.vstack([lab, line]))
    width = max(t.shape[1] for t in tiles)
    tiles = [np.hstack([t, np.full((t.shape[0], width - t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    top = np.full((head, width, 3), 255, np.uint8)
    _text(top, title, (4, 14), 0.5, INK)
    col_w = tiles[0].shape[1]
    x = 0
    w0 = _tile_h(rows[0][0], tile_h).shape[1]
    for k, name in enumerate(("image", "truth (white = terrain)", "P(terrain)",
                              "prediction - truth (red missed, blue false)")):
        _text(top, name, (x + 2, 30), 0.42, MUTED)
        x += w0 + gap
    body = np.vstack(sum([[t, np.full((gap, width, 3), 255, np.uint8)] for t in tiles], []))
    return np.vstack([top, body]), ious


# ------------------------------------------------ curves (separate process)
def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_curves_file(debug_dir: PathLike) -> Path:
    """Render ``curves.png`` from ``log.csv`` / ``val_preview.csv``.  Imports matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    d = Path(debug_dir)
    rows = _read_csv(d / "log.csv")
    if not rows:
        raise FileNotFoundError(f"no rows in {d / 'log.csv'}")
    f = lambda k: np.array([float(r.get(k) or "nan") for r in rows])
    s, n_iter = f("step"), int(float(rows[-1]["n_iter"]))
    val = []
    if (d / "val_preview.csv").is_file():
        with (d / "val_preview.csv").open(newline="") as fh:
            val = [(float(r[2]), float(r[3])) for r in csv.reader(fh) if len(r) >= 4]

    fig = Figure(figsize=(12, 7), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.subplots(2, 2).ravel()
    ax[0].plot(s, f("loss_interval"), color=C_TRAIN, lw=2)
    ax[0].set_title("loss (train, mean over interval)", fontsize=9, color=INK_HEX, loc="left")
    ax[1].plot(s, f("iou_interval"), color=C_TRAIN, lw=2, label="train batches")
    if val:
        vs, vi = np.array(val).T
        ax[1].plot(vs, vi, color=C_VAL, lw=2, marker="o", ms=5, label="held-out preview")
        ax[1].annotate(f"{vi[-1]:.3f}", (vs[-1], vi[-1]), xytext=(4, -10), textcoords="offset points",
                       fontsize=8, color=INK_HEX)
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    ax[1].set_title("IoU at inference threshold", fontsize=9, color=INK_HEX, loc="left")
    ax[2].plot(s, f("lr"), color=C_TRAIN, lw=2)
    ax[2].set_title("learning rate", fontsize=9, color=INK_HEX, loc="left")
    ax[3].plot(s, f("aspp_pre_bn_peak"), color=C_TRAIN, lw=2)
    ax[3].axhline(65504, color=MUTED_HEX, lw=1, ls="--")
    ax[3].text(s[0], 65504, " fp16 limit", va="bottom", fontsize=7, color=MUTED_HEX)
    ax[3].set_yscale("log")
    ax[3].set_title("pre-BN ASPP peak (running max) — irrelevant in bf16", fontsize=9, color=INK_HEX, loc="left")
    for a in ax:
        a.grid(True, color=GRID_HEX, lw=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(MUTED_HEX)
        a.tick_params(colors=MUTED_HEX, labelsize=8)
        a.set_xlabel("iteration", fontsize=8, color=MUTED_HEX)
        for e in range(1, int(np.nanmax(s) // n_iter) + 1):
            a.axvline(e * n_iter, color=GRID_HEX, lw=1.5)
    ax[0].set_ylim(bottom=0)                            # loss and IoU axes start at zero (v0p6)
    ax[1].set_ylim(0, 1.02)
    fig.tight_layout()
    tmp = d / "curves.tmp.png"
    fig.savefig(tmp)
    tmp.replace(d / "curves.png")                       # atomic: a viewer never sees a half-written file
    return d / "curves.png"


# -------------------------------------------------------- run housekeeping
_RUN_FILES = ("log.csv", "val_preview.csv", "curves.png", "curves_render.log", "viewer.log")


def archive_previous_run(d: Path) -> Optional[Path]:
    """
    A debug folder left by an earlier (e.g. crashed or restarted) run is moved to
    ``<name>_prev_<YYYYmmdd-HHMMSS>`` (time of its last log write), so each run's
    CSV and curves start clean.  Nothing is deleted.
    """
    import datetime as dt
    if not d.is_dir():
        return None
    used = [d / f for f in _RUN_FILES if (d / f).exists()] + list(d.glob("batch/*.png")) + list(d.glob("val/*.png"))
    if not used:
        return None
    stamp = dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in used)).strftime("%Y%m%d-%H%M%S")
    dest = d.with_name(f"{d.name}_prev_{stamp}")
    k = 1
    while dest.exists():
        dest, k = d.with_name(f"{d.name}_prev_{stamp}_{k}"), k + 1
    try:
        d.rename(dest)
    except OSError:                  # Windows: a file is open (e.g. curves.png in Photos) -> move files instead
        dest = d / f"_prev_{stamp}"
        for p in used:
            target = dest / p.relative_to(d)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.replace(target)
            except OSError:
                pass                 # still open: left in place, will be overwritten by this run
    return dest


def latest_images(d: PathLike) -> Dict[str, Optional[Path]]:
    """Newest finished curves / held-out / batch PNGs (ignores *.tmp.png)."""
    d = Path(d)
    pick = lambda sub: max((p for p in (d / sub).glob("*.png") if ".tmp." not in p.name),
                           key=lambda p: p.name, default=None)
    c = d / "curves.png"
    return {"curves": c if c.is_file() else None, "val": pick("val"), "batch": pick("batch")}


def watch(debug_dir: PathLike, interval_s: float = 5.0) -> None:
    """
    Live viewer (own process, own window): curves + newest batch panel on the
    left, newest held-out panel on the right, refreshed every ``interval_s``.
    Run by ``train(..., live=True)``, or by hand from any terminal::

        python -m mppp.mask.monitor watch checkpoints/<name>_debug
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    d = Path(debug_dir)
    fig = plt.figure(figsize=(17, 9.5))
    try:
        fig.canvas.manager.set_window_title(f"MPPP training - {d.name}")
    except Exception:
        pass
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.25, 1])
    axes = {"curves": fig.add_subplot(gs[0, 0]), "batch": fig.add_subplot(gs[1, 0]), "val": fig.add_subplot(gs[:, 1])}
    shown: Dict[str, tuple] = {}
    for a in axes.values():
        a.axis("off")
    fig.tight_layout()
    while plt.fignum_exists(fig.number):
        for k, p in latest_images(d).items():
            if p is None:
                continue
            try:
                key = (p.name, p.stat().st_mtime)
                if shown.get(k) != key:
                    img = mpimg.imread(str(p))
                    axes[k].clear(); axes[k].imshow(img); axes[k].axis("off")
                    axes[k].set_title(p.name if k != "curves" else "", fontsize=8, color=MUTED_HEX, loc="left")
                    shown[k] = key
            except Exception:                                # file being replaced; try again next tick
                continue
        fig.canvas.draw_idle()
        plt.pause(interval_s)


# ------------------------------------------------------------------ monitor
class TrainingMonitor:
    def __init__(self, out_dir: PathLike, threshold: float = 0.4, live: bool = False, tile_h: int = 300):
        self.dir = Path(out_dir)
        self.archived = archive_previous_run(self.dir)
        if self.archived:
            print(f"[monitor] previous run's debug files moved to {self.archived.name}")
        (self.dir / "batch").mkdir(parents=True, exist_ok=True)
        (self.dir / "val").mkdir(parents=True, exist_ok=True)
        self.threshold, self.live, self.tile_h = threshold, live, tile_h
        self._csv = self.dir / "log.csv"
        self._child: Optional[subprocess.Popen] = None
        self._pending = False
        self._viewer: Optional[subprocess.Popen] = None
        if live:
            self._start_viewer()

    # ---- called by train() every print interval
    def on_progress(self, rec: Dict) -> None:
        row = {k: rec[k] for k in ("epoch", "iter", "n_iter", "loss", "iou", "loss_interval", "iou_interval",
                                   "grad_norm", "lr", "aspp_pre_bn_peak", "elapsed_s") if k in rec}
        row["step"] = (rec["epoch"] - 1) * rec["n_iter"] + rec["iter"]
        new = not self._csv.exists()
        with self._csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        mosaic, _ = render_rows([(rec["image0"], rec["gt0"], rec["prob0"], "train batch")],
                                f"train  ep {rec['epoch']}  it {rec['iter']}/{rec['n_iter']}  "
                                f"loss {rec.get('loss_interval', rec['loss']):.4f}", self.threshold, self.tile_h)
        self._write(self.dir / "batch" / f"ep{rec['epoch']}_it{rec['iter']:05d}.png", mosaic)
        self._spawn_curves()

    # ---- called by train() on the fixed held-out preview set
    def on_val_preview(self, epoch: int, it: int, n_iter: int, samples: Sequence[tuple]) -> float:
        mosaic, ious = render_rows(samples, f"held-out preview  ep {epoch}  it {it}/{n_iter}",
                                   self.threshold, self.tile_h)
        self._write(self.dir / "val" / f"ep{epoch}_it{it:05d}.png", mosaic)
        m = float(np.mean(ious))
        with (self.dir / "val_preview.csv").open("a", newline="") as f:
            csv.writer(f).writerow([epoch, it, (epoch - 1) * n_iter + it, f"{m:.5f}"] + [f"{v:.5f}" for v in ious])
        self._spawn_curves()
        return m

    def finish(self, timeout: float = 120.0) -> None:
        """Wait for the curve renderer and draw the final curves."""
        if self._child is not None:
            try:
                self._child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._child.kill()
        self._child = None
        self._spawn_curves()
        if self._child is not None:
            try:
                self._child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._child.kill()

    # ---- internals
    @staticmethod
    def _write(path: Path, bgr: np.ndarray) -> None:
        import cv2
        tmp = path.with_name(path.stem + ".tmp.png")
        if cv2.imwrite(str(tmp), bgr):
            tmp.replace(path)                                # viewers never see a half-written file
        else:
            print(f"[monitor] could not write {path}")

    def _spawn_curves(self) -> None:
        if self._child is not None and self._child.poll() is None:
            self._pending = True                             # renderer busy; redraw next time
            return
        self._pending = False
        log = open(self.dir / "curves_render.log", "a")
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        try:
            self._child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "curves", str(self.dir)],
                                           stdout=log, stderr=log, creationflags=flags)
        except Exception as e:                               # plotting must never stop training
            print(f"[monitor] curve renderer not started: {e}")
            self._child = None
        finally:
            log.close()

    def _start_viewer(self) -> None:
        """Live window = a separate Python process running ``watch`` (matplotlib lives there, not here)."""
        try:
            self._viewer = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", str(self.dir)],
                                            stdout=open(self.dir / "viewer.log", "a"), stderr=subprocess.STDOUT)
            print("[monitor] live viewer started in its own window (close it any time; training is unaffected)")
        except Exception as e:
            print(f"[monitor] live viewer not started: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "curves":
        print(plot_curves_file(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "watch":
        watch(sys.argv[2])
    else:
        print("usage: python monitor.py curves|watch <debug_dir>")
        sys.exit(2)
