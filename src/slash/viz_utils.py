import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .utils import count_edges_in_prompt

# ============================================================
#   Plotting Helpers (for scoring.py)
# ============================================================

def plot_perhead_scores_and_layer_mean(
    task_name: str,
    avg_scores: np.ndarray,
    valid_mask: np.ndarray,
    out_dir: str,
    sim_metric: str,
    binarize_method: str,
    plot_cfg: dict,
):
    """
    绘制：
      1) per-head heatmap (layer x head)
      2) 基于 per-head 平均的 per-layer 折线图
    """
    num_layers, num_heads = avg_scores.shape
    scores_plot = np.where(valid_mask, avg_scores, np.nan)

    dpi = plot_cfg.get("dpi", 250)
    line_color = plot_cfg.get("line_color", "C0")

    # heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(scores_plot, aspect="auto", cmap="coolwarm", origin="lower")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.xticks(np.arange(num_heads))
    plt.yticks(np.arange(num_layers))
    plt.title(f"{task_name} - per-head avg scores ({sim_metric}, {binarize_method})")
    cbar = plt.colorbar(im)
    cbar.set_label("score")
    plt.tight_layout()
    heatmap_path = os.path.join(
        out_dir,
        f"{task_name}_perhead_heatmap.png"
    )
    plt.savefig(heatmap_path, dpi=dpi)
    plt.close()
    print(f"  -> Heatmap saved to {heatmap_path}")

    # per-layer mean line from per-head
    layer_means = np.nanmean(scores_plot, axis=1)
    x = np.arange(num_layers)
    plt.figure(figsize=(8, 4))
    plt.plot(x, layer_means, marker="o", color=line_color, linewidth=1.5)
    plt.title(f"{task_name} - per-layer avg (from per-head) ({sim_metric}, {binarize_method})")
    plt.xlabel("Layer")
    plt.ylabel("Average score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    layer_line_path = os.path.join(
        out_dir,
        f"{task_name}_perlayer_from_perhead_lineplot.png"
    )
    plt.savefig(layer_line_path, dpi=dpi)
    plt.close()
    print(f"  -> Per-layer (from per-head) line plot saved to {layer_line_path}")


def plot_perlayer_scores(
    task_name: str,
    avg_scores_layer: np.ndarray,
    out_dir: str,
    sim_metric: str,
    binarize_method: str,
    plot_cfg: dict,
):
    """
    绘制 per-layer 折线图。
    """
    num_layers = avg_scores_layer.shape[0]
    dpi = plot_cfg.get("dpi", 250)
    line_color = plot_cfg.get("line_color", "C0")

    x = np.arange(num_layers)
    plt.figure(figsize=(8, 4))
    plt.plot(x, avg_scores_layer, marker="o", color=line_color, linewidth=1.5)
    plt.title(f"{task_name} - per-layer avg scores ({sim_metric}, {binarize_method})")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    lineplot_path = os.path.join(
        out_dir,
        f"{task_name}_perlayer_lineplot.png"
    )
    plt.savefig(lineplot_path, dpi=dpi)
    plt.close()
    print(f"  -> Line plot saved to {lineplot_path}")


# ============================================================
#   Plotting Helpers for raw attention visualization (used in plot.py)
# ============================================================

def choose_sample_for_task(
    df,
    min_edges: int = 80,
    max_edges: int = 100,
    input_column: str = "input_prompt",
    max_samples: int = 5,
):
    """
    从 task 的 df 中选择若干个边数在 [min_edges, max_edges] 内的样本索引。
    返回一个索引列表（长度 <= max_samples），若无则返回空列表。
    """
    idxs = []
    num_edges_list = []
    for idx, row in df.iterrows():
        prompt_tmp = row[input_column]
        num_edges = count_edges_in_prompt(prompt_tmp)
        if min_edges <= num_edges <= max_edges:
            idxs.append(idx)
            num_edges_list.append(num_edges)
        if len(idxs) >= max_samples:
            break
    return idxs, num_edges_list


def create_layer_figure(full_attn_layer,
                        avg_roi,
                        bin_mask,
                        blurred_norm,
                        title: str,
                        local_spans,
                        g_start: int,
                        g_end: int):
    """
    full_attn_layer: [S, S] 单层多头平均后的 full attention
    avg_roi:         [N, N] ROI 内多头平均后的 attention
    bin_mask:        [N, N] 二值 mask
    blurred_norm:    [N, N] 经过形态学处理/模糊 + 归一化后的 mask
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. full attention map (layer avg)
    ax_full = axes[0, 0]
    full_show = full_attn_layer.copy()
    # if full_show.shape[1] > 0:
    #     full_show[:, 0] = 0
    im0 = ax_full.imshow(full_show, cmap="coolwarm", aspect="auto")
    ax_full.set_title(f"{title} - Full (Layer Avg)")
    ax_full.axis("off")

    # 标注大三角 ROI
    big_tri_points = [
        (g_start - 0.5, g_start - 0.5),
        (g_start - 0.5, g_end + 0.5),
        (g_end + 0.5,  g_end + 0.5),
    ]
    big_tri = patches.Polygon(
        big_tri_points,
        fill=False,
        edgecolor="yellow",
        linewidth=0.7,
    )
    ax_full.add_patch(big_tri)

    cax0 = inset_axes(
        ax_full,
        width="3%",
        height="40%",
        loc="upper right",
        borderpad=0,
    )
    fig.colorbar(im0, cax=cax0)

    # 2. 平均 ROI heatmap
    ax_raw = axes[0, 1]
    im_raw = ax_raw.imshow(avg_roi, cmap="coolwarm", aspect="auto")
    ax_raw.set_title("ROI - Layer Avg")
    ax_raw.axis("off")

    cax_raw = inset_axes(
        ax_raw,
        width="3%",
        height="40%",
        loc="upper right",
        borderpad=0,
    )
    fig.colorbar(im_raw, cax=cax_raw)

    # 3. 0/1 bin_mask
    ax_bin = axes[1, 0]
    ax_bin.imshow(bin_mask, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    ax_bin.set_title("ROI - Binary (0/1)")
    ax_bin.axis("off")

    # 4. 模糊/形态学后的 mask
    ax_blur = axes[1, 1]
    ax_blur.imshow(blurred_norm, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    ax_blur.set_title("ROI - Denoised")
    ax_blur.axis("off")

    # 每个局部 span 画一个三角形框
    for (s, e) in local_spans:
        tri_points = [
            (s - 0.5, s - 0.5),
            (s - 0.5, e + 0.5),
            (e + 0.5, e + 0.5),
        ]
        for ax_ in [ax_raw, ax_bin, ax_blur]:
            tri = patches.Polygon(
                tri_points,
                fill=False,
                edgecolor="lime",
                linewidth=0.5,
            )
            ax_.add_patch(tri)

    return fig


def create_head_figure(full_attn_head,
                       roi,
                       bin_mask,
                       blurred_norm,
                       title: str,
                       local_spans,
                       g_start: int,
                       g_end: int):
    """
    与 create_layer_figure 类似，但针对单个 head。
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. full attention map (head)
    ax_full = axes[0, 0]
    full_show = full_attn_head.copy()
    if full_show.shape[1] > 0:
        full_show[:, 0] = 0
    im0 = ax_full.imshow(full_show, cmap="coolwarm", aspect="auto")
    ax_full.set_title(f"{title} - Full (Head)")
    ax_full.axis("off")

    big_tri_points = [
        (g_start - 0.5, g_start - 0.5),
        (g_start - 0.5, g_end + 0.5),
        (g_end + 0.5,  g_end + 0.5),
    ]
    big_tri = patches.Polygon(
        big_tri_points,
        fill=False,
        edgecolor="yellow",
        linewidth=0.7,
    )
    ax_full.add_patch(big_tri)

    # 2. ROI heatmap
    ax_raw = axes[0, 1]
    im_raw = ax_raw.imshow(roi, cmap="coolwarm", aspect="auto")
    ax_raw.set_title("ROI - Head")
    ax_raw.axis("off")

    cax = inset_axes(
        ax_raw,
        width="3%",
        height="40%",
        loc="upper right",
        borderpad=0,
    )
    fig.colorbar(im_raw, cax=cax)

    # 3. 0/1 bin_mask
    ax_bin = axes[1, 0]
    ax_bin.imshow(bin_mask, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    ax_bin.set_title("ROI - Binary (0/1)")
    ax_bin.axis("off")

    # 4. 模糊/形态学后的 mask
    ax_blur = axes[1, 1]
    ax_blur.imshow(blurred_norm, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    ax_blur.set_title("ROI - Denoised")
    ax_blur.axis("off")

    for (s, e) in local_spans:
        tri_points = [
            (s - 0.5, s - 0.5),
            (s - 0.5, e + 0.5),
            (e + 0.5, e + 0.5),
        ]
        for ax_ in [ax_raw, ax_bin, ax_blur]:
            tri = patches.Polygon(
                tri_points,
                fill=False,
                edgecolor="lime",
                linewidth=0.5,
            )
            ax_.add_patch(tri)

    return fig