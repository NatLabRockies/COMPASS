"""COMPASS validation utility functions"""

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.offsetbox import HPacker
from matplotlib.offsetbox import TextArea
from matplotlib.offsetbox import VPacker
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def compute_stats(data, score_cats, truth_labels_col, score_col="Score"):
    cm = confusion_matrix(
        data[truth_labels_col].to_numpy(),
        data[score_col].isin(
            score_cats["exists_report"] | score_cats["dne_report"]
        ),
    )[::-1, ::-1]

    __, r, *___ = precision_recall_fscore_support(
        data[truth_labels_col].to_numpy(),
        data[score_col].isin(
            score_cats["exists_report"] | score_cats["dne_report"]
        ),
        average="binary",
        zero_division=0,
    )

    precision_data = data.copy()
    precision_data.loc[
        precision_data[score_col].isin(score_cats["exists_bad_report"]),
        truth_labels_col,
    ] = 0

    p, *__ = precision_recall_fscore_support(
        precision_data[truth_labels_col].to_numpy(),
        precision_data[score_col].isin(
            score_cats["exists_report"]
            | score_cats["dne_report"]
            | score_cats["exists_bad_report"]
        ),
        average="binary",
        zero_division=0,
    )

    acc = (
        data[score_col].isin(
            score_cats["exists_report"] | score_cats["dne_no_report"]
        )
    ).sum() / data.shape[0]

    f1 = 2 * p * r / (p + r) if p + r > 0 else float("NaN")

    return cm, acc, p, r, f1


def plot_compass_confusion_matrix_from_data(
    data,
    score_cats,
    title,
    truth_labels_col,
    out_fp=None,
    score_col="Score",
    x_label="COMPASS Retrieved Ordinance",
    y_label="Ordinance Exists",
    color_scores=False,
):
    cm, a, p, r, f1 = compute_stats(
        data, score_cats, truth_labels_col, score_col=score_col
    )

    num_ords = data.shape[0]

    tp = cm[0][0]
    tp = f"{tp} ({tp / num_ords:.2%})"

    fn = cm[0][1]

    fn_nf = data[score_col].isin(score_cats["exists_no_report"]).sum()
    fn_ir = data[score_col].isin(score_cats["exists_bad_report"]).sum()
    fn = (
        f"{fn_nf} ({fn_nf / num_ords:.2%}) - Not Found\n"
        f"{fn_ir} ({fn_ir / num_ords:.2%}) - Incorrectly Reported"
    )
    fp = cm[1][0]
    fp = f"{fp} ({fp / num_ords:.2%})"

    tn = cm[1][1]
    tn = f"{tn} ({tn / num_ords:.2%})"

    table_data = [["", "Yes", "No"], ["Yes", tp, fn], ["No", fp, tn]]

    plot_compass_confusion_matrix(
        table_data,
        a,
        p,
        r,
        f1,
        title,
        num_ords=len(data),
        out_fp=out_fp,
        x_label=x_label,
        y_label=y_label,
        color_scores=color_scores,
    )


def plot_compass_confusion_matrix(
    table_data,
    accuracy,
    precision,
    recall,
    f1_score,
    title,
    num_ords=None,
    out_fp=None,
    x_label="COMPASS Retrieved Ordinance",
    y_label="Ordinance Exists",
    color_scores=False,
):
    __, ax = plt.subplots(figsize=(8, 5))
    ax.set_axis_off()

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.06, 0.6, 0.6],
    )

    header_color = "#e7ecf4"
    true_pos_color = "#d8f3dc"
    true_neg_color = "#d8f3dc"
    false_pos_color = "#ffccd5"
    false_neg_color = "#fff3b0"
    default_bg = "#f0f4fa"

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 5.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(0.1)

        if col == 0:
            cell.set_width(0.1)

        if row == 0 and col == 0:
            pass

        elif row == 0 or col == 0:
            cell.set_text_props(fontsize=14)
            cell.set_facecolor(header_color)

        elif row == 1 and col == 1:
            cell.set_facecolor(true_pos_color)

        elif row == 2 and col == 2:  # ruff:ignore[magic-value-comparison]
            cell.set_facecolor(true_neg_color)

        elif row == 2 and col == 1:  # ruff:ignore[magic-value-comparison]
            cell.set_facecolor(false_pos_color)

        elif row == 1 and col == 2:  # ruff:ignore[magic-value-comparison]
            cell.set_facecolor(false_neg_color)

        else:
            cell.set_facecolor(default_bg)

    if num_ords:
        title = f"{title} (N={num_ords:,d})"

    ax.text(
        -0.2,
        0.92,
        title,
        va="center",
        ha="left",
        fontsize=16,
        transform=ax.transAxes,
        weight="bold",
    )

    ax.text(
        -0.2,
        0.45,
        y_label,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=14,
        transform=ax.transAxes,
    )
    ax.text(
        0.55,
        0.8,
        x_label,
        va="center",
        ha="center",
        fontsize=14,
        transform=ax.transAxes,
    )

    if color_scores:
        score_text = VPacker(
            children=[
                HPacker(
                    children=[
                        _metric_text("Accuracy: ", accuracy),
                        _metric_text("       F1: ", f1_score),
                    ],
                    align="baseline",
                    pad=0,
                    sep=0,
                ),
                HPacker(
                    children=[
                        _metric_text("Precision: ", precision),
                        _metric_text(" Recall: ", recall),
                    ],
                    align="baseline",
                    pad=0,
                    sep=0,
                ),
            ],
            align="left",
            pad=2,
            sep=5,
        )
        ax.add_artist(
            AnchoredOffsetbox(
                loc="center left",
                child=score_text,
                frameon=False,
                bbox_to_anchor=(0.6, 0.15),
                bbox_transform=ax.transAxes,
                borderpad=0,
                pad=0,
            )
        )
    else:
        ax.text(
            0.6,
            0.15,
            (
                # f"Accuracy: {accuracy:.2%}       F1: {f1_score:.2%}\n"
                # f"Precision: {precision:.2%} Recall: {recall:.2%}"
                # f"Precision: {precision:.2%}\n     "
                # f"Recall: {recall:.2%}"
                f"Accuracy: $\\mathbf{{{accuracy * 100:.2f}\\%}}$       "
                f"F1: $\\mathbf{{{f1_score * 100:.2f}\\%}}$\n"
                f"Precision: $\\mathbf{{{precision * 100:.2f}\\%}}$ "
                f"Recall: $\\mathbf{{{recall * 100:.2f}\\%}}$"
            ),
            va="center",
            ha="left",
            fontsize=12,
            transform=ax.transAxes,
        )

    plt.tight_layout()
    if out_fp:
        plt.savefig(out_fp, bbox_inches="tight", dpi=300)
    plt.show()


def _score_color(score):
    """Return a red-orange-green color for a score"""
    stops = (
        (0.6, to_rgb("#d62728")),
        (0.8, to_rgb("#ff7f0e")),
        (0.9, to_rgb("#2ca02c")),
    )
    score = min(max(score, stops[0][0]), stops[-1][0])

    for (lower_score, lower_color), (upper_score, upper_color) in zip(
        stops, stops[1:], strict=False
    ):
        if score <= upper_score:
            fraction = (score - lower_score) / (upper_score - lower_score)
            color = tuple(
                lower + fraction * (upper - lower)
                for lower, upper in zip(lower_color, upper_color, strict=True)
            )
            return to_hex(color)

    return to_hex(stops[-1][1])


def _metric_text(label, score):
    """Return packed regular label text and a colored bold score"""
    return HPacker(
        children=[
            TextArea(label, textprops={"fontsize": 12}),
            TextArea(
                f"{score:.2%}",
                textprops={
                    "color": _score_color(score),
                    "fontsize": 12,
                    "fontweight": "bold",
                },
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
