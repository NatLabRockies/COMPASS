"""COMPASS validation utility functions"""

import matplotlib.pyplot as plt
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
    )

    acc = (
        data[score_col].isin(
            score_cats["exists_report"] | score_cats["dne_no_report"]
        )
    ).sum() / data.shape[0]

    return cm, acc, p, r


def plot_compass_confusion_matrix_from_data(
    data, score_cats, title, truth_labels_col, out_fp=None, score_col="Score"
):
    cm, a, p, r = compute_stats(
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
        table_data, a, p, r, title, num_ords=len(data), out_fp=out_fp
    )


def plot_compass_confusion_matrix(
    table_data, accuracy, precision, recall, title, num_ords=None, out_fp=None
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

        elif row == 2 and col == 2:  # noqa: PLR2004
            cell.set_facecolor(true_neg_color)

        elif row == 2 and col == 1:  # noqa: PLR2004
            cell.set_facecolor(false_pos_color)

        elif row == 1 and col == 2:  # noqa: PLR2004
            cell.set_facecolor(false_neg_color)

        else:
            cell.set_facecolor(default_bg)

    if num_ords:
        title = f"{title} (N={num_ords:,d})"

    ax.text(
        -0.2,
        1,
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
        "Ordinance Exists",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=14,
        transform=ax.transAxes,
    )
    ax.text(
        0.55,
        0.8,
        "COMPASS Retrieved Ordinance",
        va="center",
        ha="center",
        fontsize=14,
        transform=ax.transAxes,
    )

    ax.text(
        1,
        0.15,
        (
            f"Accuracy: {accuracy:.2%}\n"
            f"Precision: {precision:.2%}\n     "
            f"Recall: {recall:.2%}"
        ),
        va="center",
        ha="center",
        fontsize=12,
        transform=ax.transAxes,
    )

    plt.tight_layout()
    if out_fp:
        plt.savefig(out_fp, bbox_inches="tight", dpi=300)
    plt.show()
