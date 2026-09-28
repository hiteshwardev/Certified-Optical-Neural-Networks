"""Single-panel journal figures with every label outside the plotting area.

Each figure is one axes on a 16 cm wide canvas, the text width of the
manuscript, so figures are embedded at their true size. The axes rectangle is
fitted to the measured extent of the tick labels, axis labels and legend
(measure, then fit), and save() refuses to write a figure in which any text
touches the plotting area, leaves the canvas, or overlaps other text.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, NullLocator  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
WIDTH_CM = 16.0
DPI = 600

# Okabe-Ito colors, ordered so that neighboring series stay separable under
# the common color-vision deficiencies. Black is reserved for reference curves.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
INK = "#000000"
LINESTYLES = ("-", "--", "-.", ":", (0, (6, 1.5, 1, 1.5, 1, 1.5)), (0, (1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "P")

RC = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.labelpad": 6.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 5.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2,
    "ytick.minor.size": 2.2,
    "xtick.major.pad": 4.0,
    "ytick.major.pad": 4.0,
    "legend.frameon": False,
    "legend.handlelength": 2.8,
    "legend.labelspacing": 0.6,
    "figure.constrained_layout.use": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "standard",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def style(i: int, marker: bool = False) -> dict:
    """Color, line style and (optionally) marker of the i-th data series."""
    out = {"color": PALETTE[i], "linestyle": LINESTYLES[i]}
    if marker:
        out["marker"] = MARKERS[i]
    return out


def figure(height_cm: float = 9.0):
    """New single-panel figure; the axes position is fixed later by save()."""
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(WIDTH_CM / 2.54, height_cm / 2.54))
    ax = fig.add_axes([0.15, 0.15, 0.55, 0.75])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def finish_axes(ax, xlog: bool = False, ylog: bool = False):
    """Log scales where requested, and minor ticks on every linear axis."""
    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def log_axis(ax, ticks, base: int = 10):
    """Logarithmic x axis with major ticks at the given values, labeled as plain numbers."""
    ax.set_xscale("log", base=base)
    ax.set_xticks(list(ticks), [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def legend(ax, where: str = "right", ncol: int = 1, **kwargs):
    """Legend outside the axes, either in a column on the right or above."""
    if where == "right":
        return ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0),
                         borderaxespad=0.0, **kwargs)
    return ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=ncol,
                     borderaxespad=0.0, columnspacing=1.6, **kwargs)


def _text_boxes(fig, ax, renderer):
    """(name, bbox) for every visible text element of the figure."""
    boxes = []
    for name, artist in (("x label", ax.xaxis.label), ("y label", ax.yaxis.label),
                         ("title", ax.title)):
        if artist.get_visible() and artist.get_text().strip():
            boxes.append((name, artist.get_window_extent(renderer)))
    for axis, tag in ((ax.xaxis, "x tick"), (ax.yaxis, "y tick")):
        lo, hi = sorted(axis.get_view_interval())
        for tick in axis.get_major_ticks():
            label = tick.label1
            if (label.get_visible() and label.get_text().strip()
                    and lo - 1e-12 * max(1.0, abs(hi)) <= tick.get_loc() <= hi * (1 + 1e-12)):
                boxes.append((f"{tag} {label.get_text()}", label.get_window_extent(renderer)))
    leg = ax.get_legend()
    if leg is not None:
        boxes.append(("legend", leg.get_window_extent(renderer)))
    for text in fig.texts + ax.texts:
        if text.get_visible() and text.get_text().strip():
            boxes.append((f"text '{text.get_text()[:20]}'", text.get_window_extent(renderer)))
    return boxes


def _overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def fit(fig, ax, pad_pt: float = 5.0, iterations: int = 4):
    """Place the axes so that its labels, ticks and legend just fit the canvas."""
    pad = pad_pt * fig.dpi / 72.0
    for _ in range(iterations):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.bbox.width, fig.bbox.height
        box = ax.get_window_extent(renderer)
        tight = ax.get_tightbbox(renderer)
        x0 = (box.x0 - tight.x0 + pad) / width
        x1 = 1.0 - (tight.x1 - box.x1 + pad) / width
        y0 = (box.y0 - tight.y0 + pad) / height
        y1 = 1.0 - (tight.y1 - box.y1 + pad) / height
        ax.set_position([x0, y0, x1 - x0, y1 - y0])


def check(fig, ax, schematic: bool = False) -> list:
    """Layout problems: text inside the plotting area, off the canvas, or overlapping."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    boxes = _text_boxes(fig, ax, renderer)
    problems = []
    for name, b in boxes:
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > width + 0.5 or b.y1 > height + 0.5:
            problems.append(f"{name} leaves the canvas")
    if not schematic:
        area = ax.get_window_extent(renderer)
        for name, b in boxes:
            if _overlap(b, area) > 0.5:
                problems.append(f"{name} intrudes into the plotting area")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i][1], boxes[j][1]) > 0.5:
                problems.append(f"{boxes[i][0]} overlaps {boxes[j][0]}")
    return problems


def save(fig, ax, name: str, schematic: bool = False) -> Path:
    """Fit the layout, verify it, and write figures/<name>.pdf and .png."""
    if not schematic:
        fit(fig, ax)
    problems = check(fig, ax, schematic=schematic)
    if problems:
        plt.close(fig)
        raise RuntimeError(f"layout check failed for {name}: " + "; ".join(problems))
    FIGURES.mkdir(exist_ok=True)
    pdf, png = FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return png
