"""
Generic Matplotlib plotting helpers.

Supports:
- 2 parameters: x, y  -> single line plot
- 4 parameters: x1, y1, x2, y2 -> two lines on the same axes

Examples:
    from plot_graphs import plot_graph

    plot_graph(x, y, title="Loss", xlabel="Epoch", ylabel="Loss")

    plot_graph(x1, y1, x2, y2,
               labels=("train", "val"),
               title="Accuracy", xlabel="Epoch", ylabel="Acc")
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt

Number = Union[int, float]

def _to_list(a: Iterable[Number]) -> list[Number]:
    return list(a)

def plot_graph(
    x: Sequence[Number],
    y: Sequence[Number],
    x2: Optional[Sequence[Number]] = None,
    y2: Optional[Sequence[Number]] = None,
    *,
    labels: Tuple[str, str] = ("Series 1", "Series 2"),
    title: str = "Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    grid: bool = True,
    marker: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot one series (x,y) or two series (x,y,x2,y2).

    Args:
        x, y: First series data.
        x2, y2: Optional second series data. If provided, both must be provided.
        labels: Legend labels for (series1, series2).
        title, xlabel, ylabel: Plot text.
        grid: Show grid.
        marker: Optional marker style, e.g. "o", ".", "x".
        figsize: Figure size.
        save_path: If set, saves the figure to this path (e.g. "plot.png").
        show: If True, calls plt.show().
    """
    if (x2 is None) != (y2 is None):
        raise ValueError("Provide both x2 and y2 for a second series, or neither.")

    x = _to_list(x)
    y = _to_list(y)

    if len(x) != len(y):
        raise ValueError(f"x and y lengths must match. Got len(x)={len(x)} len(y)={len(y)}")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, label=labels[0], marker=marker)

    if x2 is not None and y2 is not None:
        x2 = _to_list(x2)
        y2 = _to_list(y2)
        if len(x2) != len(y2):
            raise ValueError(f"x2 and y2 lengths must match. Got len(x2)={len(x2)} len(y2)={len(y2)}")
        ax.plot(x2, y2, label=labels[1], marker=marker)
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    # Demo
    epochs = list(range(1, 11))
    train_loss = [1.0, 0.9, 0.82, 0.75, 0.7, 0.66, 0.63, 0.61, 0.6, 0.59]
    val_loss =   [1.1, 1.0, 0.95, 0.9, 0.88, 0.87, 0.86, 0.87, 0.89, 0.92]

    plot_graph(
        epochs, train_loss,
        epochs, val_loss,
        labels=("train_loss", "val_loss"),
        title="Loss vs Epoch",
        xlabel="Epoch",
        ylabel="Loss",
        marker="o",
    )