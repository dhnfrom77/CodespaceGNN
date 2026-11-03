#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 1: Kiểm toán Tập dữ liệu `calib_123.csv` (10,000 Cặp Tham số), Định량 Hóa Lặp giả
        Tiến hóa qua Mô hình Cơ sở Cây Hình Sao (Star Baseline M0) và Trực quan hóa Figure 0A & 0B-C.

Tham chiếu:
  - Figure 0A: Toàn cảnh đồ thị cây phả hệ siêu dài 148 loài kiến.
  - Figure 0B-C: Biểu đồ ngang phân bố trọng số cắt tỉa gốc w_{phylolm, i} và tính trạng lá X_{0, i}.
  - Tiểu luận Mục 2.3 - 2.4 & Cell 4 - Cell 7 trong Master Notebook.
"""

# 1. Thư viện chuẩn của Python
import json
import math
from pathlib import Path
from typing import Dict, Tuple

# 2. Thư viện bên thứ ba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 3. Mã của chính dự án
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_gnn.config_paths import (
    CALIB_CSV_PATH,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    ensure_output_directories,
)
from bm_gnn.step0_tree_and_bm_simulator import (
    FromScratchPhylogenyBuilder,
    c4_factor,
    compute_reusable_metrics,
)


def run_star_baseline(t_diag: np.ndarray, x_mat: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Tính bộ ước lượng ngây thơ M0 (Star Phylogeny Baseline — giả định S = I_n):
      - mu_hat_star = (1/n) * sum_i x_i
      - sigma_hat_star_reml = sqrt(Q_star / (n - 1)) / c_4(n - 1)
    """
    n_leaves = len(t_diag)
    df = n_leaves - 1
    c4_val = c4_factor(df)
    w_star = (1.0 / t_diag) / np.sum(1.0 / t_diag)
    mu_hat = x_mat @ w_star
    res = x_mat - mu_hat[:, None]
    q_star = np.sum((res ** 2) / t_diag[None, :], axis=1)
    sigma_mle = np.sqrt(q_star / n_leaves)
    sigma_reml = np.sqrt(q_star / df) / c4_val
    return {
        "mu_hat": mu_hat,
        "sigma_mle": sigma_mle,
        "sigma_reml": sigma_reml,
        "Q_star": q_star,
        "w_star": w_star,
    }


def load_and_audit_calib123(
    builder: FromScratchPhylogenyBuilder,
    csv_path: Path = CALIB_CSV_PATH,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Đọc tệp `calib_123.csv` (chỉ đọc từ `data/raw/`) và tách (mu_true, sigma_true, X_mat)."""
    df_calib = pd.read_csv(csv_path)
    mu_true = df_calib["mu"].values.astype(np.float64)
    sigma_true = df_calib["sigma"].values.astype(np.float64)
    x_mat = df_calib[builder.species_names].values.astype(np.float64)
    return df_calib, mu_true, sigma_true, x_mat


def plot_figure0bc_horizontal(
    builder: FromScratchPhylogenyBuilder,
    sample_traits: np.ndarray,
    sample_mu: float,
    sample_sig: float,
    show_leaf_values: bool = True,
    save_path: Path = FIGURES_DIR / "advisor_fig0_bc_horizontal_weights_and_traits.png",
) -> Path:
    """
    Vẽ biểu đồ nằm ngang `Figure 0B-C` gồm Panel (B) trọng số cắt tỉa w_{phylolm, i}
    và Panel (C) hồ sơ tính trạng lá X_{0, i} với tùy chọn hiển thị giá trị tại toàn bộ 148 nút lá.
    """
    ensure_output_directories()
    leaf_nodes = [u for u in builder.postorder_nodes if u.is_leaf()]
    leaf_nodes = sorted(leaf_nodes, key=lambda node: node.y_pos)
    x_idx = np.arange(len(leaf_nodes))
    w_ordered = np.array([builder.w_gls[u.node_id] for u in leaf_nodes])
    traits_ordered = np.array([sample_traits[u.node_id] for u in leaf_nodes])

    def clade_color(pos: int) -> str:
        if pos < 52:
            return "#1d3557"
        if pos < 77:
            return "#2a9d8f"
        if pos < 103:
            return "#e76f51"
        if pos < 130:
            return "#6a4c93"
        return "#d62828"

    colors = [clade_color(i) for i in range(len(leaf_nodes))]

    fig, (ax_b, ax_c) = plt.subplots(2, 1, figsize=(21.5, 11.5), dpi=220, sharex=True)
    fig.suptitle(
        "Figure 0B-C (Bố Cục Ngang): Phân Bố Trọng Số Cắt Tỉa Gốc Tối Ưu w_{phylolm, i} (Panel B) "
        "& Hồ Sơ Tính Trạng Lá Quan Sát X_{0, i} (Panel C) Trên 148 Loài Kiến",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    ax_b.bar(x_idx, w_ordered, color=colors, width=0.78, alpha=0.88)
    ax_b.axhline(
        1.0 / 148.0,
        color="#d62828",
        ls="--",
        lw=1.6,
        label="Trọng số đẳng hướng Star 1/148 = 0.00676",
    )
    ax_b.set_ylabel("Trọng số gốc w_{phylolm, i}")
    ax_b.set_title("(B) Trọng Số Cắt Tỉa Gốc Tối Ưu w_{phylolm, i} Dọc Theo 148 Loài Lá (Tùy chọn show_leaf_values=" + str(show_leaf_values) + ")")
    ax_b.legend(loc="upper left", fontsize=9.0)
    ax_b.grid(True, alpha=0.25)

    ax_c.bar(x_idx, traits_ordered, color=colors, width=0.78, alpha=0.88)
    ax_c.axhline(
        sample_mu,
        color="#1d3557",
        ls="-",
        lw=1.6,
        label=f"Trạng thái gốc thực tế mu_0 = {sample_mu:+.3f} (sigma_0 = {sample_sig:.3f})",
    )
    ax_c.set_ylabel("Giá trị tính trạng lá X_{0, i}")
    ax_c.set_xlabel("Chỉ số thứ tự loài lá theo nhánh cây phả hệ (0 .. 147)")
    ax_c.set_title("(C) Hồ Sơ Giá Trị Tính Trạng Quan Sát Tại 148 Nút Lá (Mẫu k = 0 trong calib_123.csv)")
    ax_c.legend(loc="upper left", fontsize=9.0)
    ax_c.grid(True, alpha=0.25)

    if show_leaf_values:
        for idx_i, (w_val, tr_val) in enumerate(zip(w_ordered, traits_ordered)):
            if idx_i % 2 == 0 or w_val > 0.02:
                ax_b.text(
                    idx_i,
                    w_val + 0.003,
                    f"{w_val:.4f}",
                    rotation=90,
                    fontsize=5.8,
                    ha="center",
                    va="bottom",
                )
                ax_c.text(
                    idx_i,
                    tr_val + (0.12 if tr_val >= 0 else -0.35),
                    f"{tr_val:+.2f}",
                    rotation=90,
                    fontsize=5.8,
                    ha="center",
                    va="bottom",
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_step1_pipeline() -> Dict[str, object]:
    """Thực thi trọn vẹn Bước 1: kiểm toán `calib_123.csv` và đánh giá Star Baseline M0."""
    ensure_output_directories()
    builder = FromScratchPhylogenyBuilder()
    df_calib, mu_true, sigma_true, x_mat = load_and_audit_calib123(builder)

    t_diag = np.diag(builder.S_mat)
    res_star = run_star_baseline(t_diag, x_mat)

    pred_se_star = res_star["sigma_reml"] * math.sqrt(
        float(builder.ones.T @ builder.S_mat @ builder.ones) / (builder.n_leaves ** 2)
        if hasattr(builder, "ones")
        else float(np.mean(builder.S_mat))
    )
    metrics_mu_star = compute_reusable_metrics(
        mu_true, res_star["mu_hat"], sigma_scale=sigma_true, pred_se=pred_se_star
    )
    metrics_sig_star = compute_reusable_metrics(
        sigma_true, res_star["sigma_reml"], sigma_scale=sigma_true
    )

    summary = {
        "n_samples_calib123": int(len(df_calib)),
        "n_leaves": int(builder.n_leaves),
        "nan_count": int(df_calib.isna().sum().sum()),
        "inf_count": int(np.isinf(df_calib.values).sum()),
        "mu_true_range": [float(mu_true.min()), float(mu_true.max())],
        "sigma_true_range": [float(sigma_true.min()), float(sigma_true.max())],
        "kappa_tree": builder.kappa_tree,
        "star_baseline_full10000": {
            "mu": metrics_mu_star,
            "sigma_reml": metrics_sig_star,
        },
    }
    return summary


if __name__ == "__main__":
    step1_info = run_step1_pipeline()
    print("===== KẾT QUẢ BƯỚC 1: KIỂM TOÁN DỮ LIỆU calib_123.csv & MÔ HÌNH STAR BASELINE (M0) =====")
    print(f"  * Số mẫu trong calib_123.csv : {step1_info['n_samples_calib123']:,d} mẫu x {step1_info['n_leaves']} loài")
    print(f"  * Kiểm tra NaN / Inf         : NaN = {step1_info['nan_count']}, Inf = {step1_info['inf_count']}")
    print(f"  * Miền giá trị mu_k          : [{step1_info['mu_true_range'][0]:+.4f}, {step1_info['mu_true_range'][1]:+.4f}]")
    print(f"  * Miền giá trị sigma_k       : [{step1_info['sigma_true_range'][0]:.4f}, {step1_info['sigma_true_range'][1]:.4f}]")
    print(f"  * Hệ số khấu hao phương sai  : kappa_T = {step1_info['kappa_tree']:.6f}")
    m_mu = step1_info["star_baseline_full10000"]["mu"]
    m_sg = step1_info["star_baseline_full10000"]["sigma_reml"]
    print(f"  * Star Baseline (M0) trên mu : MSE = {m_mu['MSE']:.6f}, NRMSE_sigma = {m_mu['NRMSE_by_sigma']:.4f}")
    print(f"  * Star Baseline (M0) trên sig: MSE = {m_sg['MSE']:.6f}, Bias = {m_sg['Bias']:+.4f}, R2 = {m_sg['R2']:.4f}")
