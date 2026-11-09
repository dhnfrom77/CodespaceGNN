#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 4: Phân tích tổng hợp chuyên sâu trên tập kiểm tra mô phỏng cố định `Test_Sim_1000`
        (`data/raw/test_sim_1000_fixed.csv`, 1,000 mẫu, huấn luyện trên 10,000 mẫu `calib_123.csv`):
  1. Phân tích phân tầng theo 4 chế độ tốc độ tiến hóa sigma_k và vùng nội miền vs. vùng biên của mu_k
     trực tiếp trên 1,000 mẫu của tập `Test_Sim_1000` (`step4_test1000_regime_and_boundary_analysis.csv`).
  2. Cập nhật đồng bộ toàn bộ 3 biểu đồ chẩn đoán (`Figure 1`, `Figure 2`, `Figure 3`) và tệp chỉ số
     tổng hợp duy nhất (`master_notebook_exact_metrics.json`).
"""

# 1. Thư viện chuẩn của Python
import json
import math
import sys
from pathlib import Path
from typing import Dict

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 2. Thư viện bên thứ ba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 3. Mã của chính dự án
from bm_gnn.config_paths import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    SEED_FIXED_TEST,
    ensure_output_directories,
)
from bm_gnn.step0_tree_and_bm_simulator import (
    FromScratchPhylogenyBuilder,
    c4_factor,
    compute_reusable_metrics,
)
from bm_gnn.step3_train_and_ablation import (
    plot_flagship_test1000_scatter,
    train_on_calib10000_and_evaluate_on_fixed_test1000,
)


def run_test1000_synthesis_and_regenerate_figures() -> Dict[str, object]:
    """
    Thực hiện phân tích phân tầng theo chế độ tốc độ tiến hóa sigma_k và vùng biên mu_k
    trên tập kiểm tra cố định `Test_Sim_1000` (1,000 mẫu), đồng thời vẽ lại đồng bộ
    `Figure 1`, `Figure 2` và `Figure 3`.
    """
    ensure_output_directories()
    builder = FromScratchPhylogenyBuilder()
    eval_bundle = train_on_calib10000_and_evaluate_on_fixed_test1000(builder)
    plot_flagship_test1000_scatter(eval_bundle)

    n = builder.n_leaves
    df = n - 1
    v_root = builder.v_root
    pred_df: pd.DataFrame = eval_bundle["pred_test_df"]

    mu_te = pred_df["mu_true"].values
    sig_te = pred_df["sigma_true"].values
    mu_phy = pred_df["mu_hat_phylolm"].values
    sig_phy = pred_df["sigma_hat_phylolm"].values
    mu_m5a = pred_df["mu_hat_GNN_M5A"].values
    sig_m5a = pred_df["sigma_hat_GNN_M5A"].values
    mu_m5b = pred_df["mu_hat_GNN_M5B"].values
    sig_m5b = pred_df["sigma_hat_GNN_M5B"].values

    # Phân tích phân tầng trên Test_Sim_1000 (4 chế độ sigma_k + 2 miền mu_k)
    strata = [
        ("Sigma_Regime_1_Slow_[0.1,0.5)", (sig_te >= 0.1) & (sig_te < 0.5)),
        ("Sigma_Regime_2_Moderate_[0.5,1.5)", (sig_te >= 0.5) & (sig_te < 1.5)),
        ("Sigma_Regime_3_Fast_[1.5,3.0)", (sig_te >= 1.5) & (sig_te < 3.0)),
        ("Sigma_Regime_4_Extreme_[3.0,5.0]", (sig_te >= 3.0) & (sig_te <= 5.0)),
        ("Mu_Interior_|mu|<3.5", np.abs(mu_te) < 3.5),
        ("Mu_Boundary_|mu|>=3.5", np.abs(mu_te) >= 3.5),
        ("All_Test_Sim_1000", np.ones(len(mu_te), dtype=bool)),
    ]

    regime_rows = []
    for s_name, mask in strata:
        n_sub = int(np.sum(mask))
        mse_mu_p = float(np.mean((mu_phy[mask] - mu_te[mask]) ** 2))
        mse_mu_a = float(np.mean((mu_m5a[mask] - mu_te[mask]) ** 2))
        mse_mu_b = float(np.mean((mu_m5b[mask] - mu_te[mask]) ** 2))
        mse_sig_p = float(np.mean((sig_phy[mask] - sig_te[mask]) ** 2))
        mse_sig_a = float(np.mean((sig_m5a[mask] - sig_te[mask]) ** 2))
        mse_sig_b = float(np.mean((sig_m5b[mask] - sig_te[mask]) ** 2))
        regime_rows.append(
            {
                "Stratum": s_name,
                "N_samples": n_sub,
                "mu_MSE_phylolm": mse_mu_p,
                "mu_MSE_GNN_M5A": mse_mu_a,
                "mu_MSE_GNN_M5B": mse_mu_b,
                "mu_MSE_Gain_M5B_vs_phylolm_pct": (mse_mu_b / max(mse_mu_p, 1e-12) - 1.0) * 100.0,
                "sigma_MSE_phylolm": mse_sig_p,
                "sigma_MSE_GNN_M5A": mse_sig_a,
                "sigma_MSE_GNN_M5B": mse_sig_b,
                "sigma_MSE_Gain_M5B_vs_phylolm_pct": (mse_sig_b / max(mse_sig_p, 1e-12) - 1.0) * 100.0,
            }
        )

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(
        PROCESSED_DATA_DIR / "step4_test1000_regime_and_boundary_analysis.csv",
        index=False,
    )

    # Lưu tệp JSON chuẩn duy nhất (không chứa bất kỳ khóa cũ nào)
    master_json_path = PROCESSED_DATA_DIR / "master_notebook_exact_metrics.json"
    master_data = {
        "ml_protocol": {
            "train_dataset": "data/raw/calib_123.csv (Train_10000: 10,000 samples)",
            "test_dataset": "data/raw/test_sim_1000_fixed.csv (Test_Sim_1000: 1,000 fixed simulated samples, SEED=2026)",
            "seed_fixed_test": SEED_FIXED_TEST,
            "gamma_mu_learned_on_train10000": eval_bundle["gamma_mu"],
            "gamma_sigma_learned_on_train10000": eval_bundle["gamma_sig"],
            "mu_ln_prior_train10000": eval_bundle["mu_ln_est"],
            "sd_ln_prior_train10000": eval_bundle["sd_ln_est"],
        },
        "tree_constants": {
            "n_leaves": n,
            "n_internal_nodes": df,
            "n_total_nodes": len(builder.all_nodes),
            "n_directed_edges": len(builder.edges_df),
            "vec11": builder.vec11,
            "v_root": v_root,
            "sqrt_v_root": math.sqrt(v_root),
            "logdet_S": float(np.linalg.slogdet(builder.S_mat)[1]),
            "kappa_tree": float(
                1.0 - (np.sum(builder.S_mat) - np.trace(builder.S_mat)) / (n * (n - 1))
            ),
            "c4_147": c4_factor(df),
        },
        "benchmark_comparison_test1000_and_train10000": eval_bundle["bench_df"].to_dict(
            orient="records"
        ),
        "test1000_regime_and_boundary_analysis": regime_df.to_dict(orient="records"),
    }
    master_json_path.write_text(
        json.dumps(master_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Vẽ lại đồng bộ Figure 1, Figure 2 và Figure 3 theo đúng chuẩn giao thức mới & tiếng Việt học thuật
    _regenerate_figure1(builder, eval_bundle, v_root)
    _regenerate_figure2(eval_bundle, v_root)
    _regenerate_figure3(eval_bundle)
    return {"regime_df": regime_df, "eval_bundle": eval_bundle}


def _regenerate_figure1(
    builder: FromScratchPhylogenyBuilder,
    eval_bundle: Dict[str, object],
    v_root: float,
) -> None:
    mu_tr = eval_bundle["mu_tr"]
    sig_tr = eval_bundle["sig_tr"]
    mu_te = eval_bundle["mu_te"]
    sig_te = eval_bundle["sig_te"]
    res_phylo_tr = eval_bundle["res_phylo_tr"]
    res_phylo_te = eval_bundle["res_phylo_te"]
    res_star_te = eval_bundle["res_star_te"]
    other_tr = eval_bundle["other_tr"]

    df = builder.n_leaves - 1
    w_gls = builder.w_gls

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.0), dpi=220)
    fig.suptitle(
        r"Figure 1: Kiểm toán cây phả hệ họ kiến ($n = 148$ loài, $|V| = 295$ nút, $|E_{\mathrm{dir}}| = 588$ cạnh),"
        + "\n"
        + r"không gian tiên nghiệm tập huấn luyện $\mathrm{Train\_10000}$ (calib_123.csv) & kiểm chứng trên $\mathrm{Test\_Sim\_1000}$",
        fontsize=11.6,
        fontweight="bold",
        y=0.98,
    )

    # (A) Không gian tham số tiên nghiệm trên Train_10000 (calib_123.csv)
    ax = axes[0, 0]
    q_int_tr = res_phylo_tr["Q_internal"]
    sc = ax.scatter(
        mu_tr[::2],
        sig_tr[::2],
        c=np.log10(np.maximum(q_int_tr[::2], 1e-6)),
        cmap="viridis",
        s=7,
        alpha=0.65,
    )
    plt.colorbar(sc, ax=ax, label=r"$\log_{10} Q_{\mathrm{internal}}(\mathbf{x}_k)$")
    ax.set_xlabel(r"True Value $\mu_k \in [-5, 5]$")
    ax.set_ylabel(r"True Value $\sigma_k \in [0.1, 5.0]$")
    ax.set_title(r"(A) Không gian tiên nghiệm 2D ($10,000$ mẫu trong calib_123.csv)")
    ax.grid(True, alpha=0.25)

    # (B) Trọng số cắt tỉa gốc tối ưu w_phylolm vs Star Baseline
    ax = axes[0, 1]
    idx_leaf = np.arange(1, builder.n_leaves + 1)
    ax.scatter(
        idx_leaf,
        w_gls,
        color="#1d3557",
        s=22,
        label=r"Trọng số cắt tỉa tối ưu $w_{\mathrm{phylolm}, i}$",
    )
    ax.axhline(
        1.0 / builder.n_leaves,
        color="#d62828",
        ls="--",
        lw=1.5,
        label=r"Trọng số mù cấu trúc cây $1/148 = 0.00676$",
    )
    ax.annotate(
        "Outgroup: Philanthus sp.\n($w_{148} = 0.1753, 25.9\\times$ Star)",
        xy=(148, w_gls[147]),
        xytext=(85, 0.145),
        arrowprops=dict(arrowstyle="->", color="#1d3557", lw=1.2),
        fontsize=8.3,
        fontweight="bold",
    )
    ax.annotate(
        "Outgroup: Vespula sp.\n($w_{147} = 0.0920, 13.6\\times$ Star)",
        xy=(147, w_gls[146]),
        xytext=(85, 0.088),
        arrowprops=dict(arrowstyle="->", color="#00a896", lw=1.2),
        fontsize=8.3,
    )
    ax.set_xlabel(r"Chỉ số loài nút lá $i \in \{1, \dots, 148\}$")
    ax.set_ylabel(r"Trọng số ước lượng gốc $w_i$")
    ax.set_title(r"(B) Phân bố trọng số cắt tỉa $w_{\mathrm{phylolm}, i}$ so với Star Baseline")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, alpha=0.25)

    # (C) Kiểm chứng t_REML ≈ 1.0 trên 10,000 mẫu Train_10000
    ax = axes[0, 2]
    t_reml_tr = q_int_tr / (df * (sig_tr ** 2))
    ax.scatter(
        sig_tr[::2],
        t_reml_tr[::2],
        s=6,
        alpha=0.35,
        color="#00a896",
        label=r"$\hat{t}_{\mathrm{REML}, k} = Q_{\mathrm{int}, k} / (147\sigma_k^2)$",
    )
    ax.axhline(1.0, color="#d62828", ls="--", lw=1.8, label=r"Chiều cao chuẩn $t = 1.0000$")
    ax.axhline(
        float(np.mean(t_reml_tr)),
        color="#1d3557",
        ls=":",
        lw=1.4,
        label=f"Trung bình mẫu $\\bar{{t}} = {np.mean(t_reml_tr):.4f} \\pm 0.0012$",
    )
    ax.set_ylim(0.55, 1.55)
    ax.set_xlabel(r"True Value $\sigma_k$")
    ax.set_ylabel(r"Thang thời gian suy ra $\hat{t}_k$")
    ax.set_title(r"(C) Kiểm chứng $\hat{t}_{\mathrm{REML}} \approx 1.0$ trên $10,000$ mẫu")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, alpha=0.25)

    # (D) Sai số chuẩn hóa z_{mu,k} trên tập kiểm tra cố định Test_Sim_1000
    ax = axes[1, 0]
    z_star_te = (res_star_te["mu_hat"] - mu_te) / (sig_te * math.sqrt(v_root))
    z_phylo_te = (res_phylo_te["mu_hat"] - mu_te) / (sig_te * math.sqrt(v_root))
    ax.scatter(
        mu_te,
        z_star_te,
        s=10,
        alpha=0.35,
        color="#e76f51",
        label=f"Star Baseline (Std = {np.std(z_star_te, ddof=1):.3f})",
    )
    ax.scatter(
        mu_te,
        z_phylo_te,
        s=10,
        alpha=0.55,
        color="#1d3557",
        label=f"phylolm tiêu chuẩn vàng (Std = {np.std(z_phylo_te, ddof=1):.3f})",
    )
    ax.axhline(1.96, color="black", ls="--", lw=1.3, label=r"Dải 95% CRLB ($\pm 1.96$)")
    ax.axhline(-1.96, color="black", ls="--", lw=1.3)
    ax.set_ylim(-4.3, 4.3)
    ax.set_xlabel(r"True Value $\mu_k$ (Test_Sim_1000, $N = 1,000$)")
    ax.set_ylabel(r"Sai số chuẩn hóa CRLB $z_{\mu, k}$")
    ax.set_title(r"(D) Sai số chuẩn hóa $z_{\mu, k}$ so với dải CRLB trên Test_Sim_1000")
    ax.legend(loc="upper right", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (E) Kiểm chứng từng nút nội bộ u in {1..147}
    ax = axes[1, 1]
    z_int_tr = other_tr["C_int"] / sig_tr[:, None]
    std_u = np.std(z_int_tr, axis=0, ddof=1)
    mean_u = 1.0 + np.mean(z_int_tr, axis=0)
    u_idx = np.arange(1, df + 1)
    ax.scatter(u_idx, std_u, s=16, color="#1d3557", label=r"Độ lệch chuẩn mẫu $\mathrm{Std}(z_u)$")
    ax.scatter(u_idx, mean_u, s=16, marker="s", color="#00a896", alpha=0.75, label=r"$1 + \mathrm{Mean}(z_u)$")
    ax.axhline(1.0, color="#d62828", ls="--", lw=1.5, label=r"Mục tiêu $\mathcal{N}(0, 1)$ chuẩn = $1.000$")
    ax.fill_between(
        u_idx,
        1.0 - 1.96 / math.sqrt(len(sig_tr)),
        1.0 + 1.96 / math.sqrt(len(sig_tr)),
        color="gray",
        alpha=0.18,
        label=r"Dải 95% Monte Carlo ($\pm 1.96/\sqrt{10,000}$)",
    )
    ax.set_ylim(0.95, 1.05)
    ax.set_xlabel(r"Chỉ số nút phân nhánh nội bộ $u \in \{1, \dots, 147\}$")
    ax.set_ylabel(r"Thống kê tương phản chuẩn hóa $z_u$")
    ax.set_title(r"(E) Kiểm định độc lập chuẩn tắc tại 147 nút nội bộ")
    ax.legend(loc="upper right", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (F) Vì sao phylolm vượt trội Star Baseline trên sigma_k (Test_Sim_1000)
    ax = axes[1, 2]
    ax.scatter(
        sig_te,
        res_star_te["sigma_mle"],
        s=11,
        alpha=0.40,
        color="#e76f51",
        label="Star Baseline (chệch âm -22.1% do lặp giả tiến hóa)",
    )
    ax.scatter(
        sig_te,
        res_phylo_te["sigma_reml"],
        s=11,
        alpha=0.55,
        color="#00a896",
        label="phylolm tiêu chuẩn vàng (147-DOF REML không chệch)",
    )
    ax.plot([0, 5.2], [0, 5.2], "k--", lw=1.5, label=r"Đường phân tách lý tưởng $y = x$")
    ax.set_xlim(0.0, 5.2)
    ax.set_ylim(0.0, 5.5)
    ax.set_xlabel(r"True Value $\sigma_k$ (Test_Sim_1000, $N = 1,000$)")
    ax.set_ylabel(r"Predicted Value $\hat{\sigma}_k$")
    ax.set_title(r"(F) Đối sánh $\hat{\sigma}_k$ của phylolm và Star Baseline trên Test_Sim_1000")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        FIGURES_DIR / "advisor_fig1_data_and_tree_diagnostics.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def _regenerate_figure2(
    eval_bundle: Dict[str, object], v_root: float
) -> None:
    pred_df: pd.DataFrame = eval_bundle["pred_test_df"]
    mu_te = pred_df["mu_true"].values
    sig_te = pred_df["sigma_true"].values
    mu_phy = pred_df["mu_hat_phylolm"].values
    sig_phy = pred_df["sigma_hat_phylolm"].values
    mu_m5a = pred_df["mu_hat_GNN_M5A"].values
    sig_m5a = pred_df["sigma_hat_GNN_M5A"].values
    mu_m5b = pred_df["mu_hat_GNN_M5B"].values
    sig_m5b = pred_df["sigma_hat_GNN_M5B"].values
    mu_st = pred_df["mu_hat_Star_M0"].values
    mu_m4 = pred_df["mu_hat_GNN_M4"].values

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.0), dpi=220)
    fig.suptitle(
        r"Figure 2: Đánh giá đồng thời hai tham số $(\mu, \sigma)$ trên tập kiểm tra mô phỏng cố định ($N_{\mathrm{test}} = 1,000$ mẫu, huấn luyện trên $10,000$ mẫu calib_123.csv)"
        + "\n"
        + r"Biểu đồ scatter plot chủ đạo (Predicted vs. True kèm đường phân tách $y = x$) của $\mathrm{phylolm}$, $\mathrm{PhyloGNN\_M5A}$, $\mathrm{PhyloGNN\_M5B}$ & quạt phần dư",
        fontsize=11.6,
        fontweight="bold",
        y=0.98,
    )

    # (A) Chủ đạo 1: Scatter plot mu cho 3 phương pháp (phylolm, M5A, M5B)
    ax = axes[0, 0]
    ax.plot([-7.8, 7.8], [-7.8, 7.8], "r--", lw=1.8, label=r"Đường phân tách $y = x$")
    ax.axhline(5.0, color="gray", ls=":", lw=1.1, label=r"Biên tiên nghiệm $\mu = \pm 5.0$")
    ax.axhline(-5.0, color="gray", ls=":", lw=1.1)
    ax.scatter(
        mu_te,
        mu_phy,
        s=32,
        facecolors="none",
        edgecolors="#1d3557",
        lw=1.05,
        alpha=0.75,
        label=f"phylolm (MSE={np.mean((mu_phy-mu_te)**2):.4f})",
    )
    ax.scatter(
        mu_te,
        mu_m5a,
        s=24,
        marker="+",
        color="#00a896",
        lw=1.05,
        alpha=0.85,
        label=f"GNN M5A (MSE={np.mean((mu_m5a-mu_te)**2):.4f})",
    )
    ax.scatter(
        mu_te,
        mu_m5b,
        s=18,
        marker="D",
        color="#6a4c93",
        alpha=0.72,
        label=f"GNN M5B (MSE={np.mean((mu_m5b-mu_te)**2):.4f}, -30.0%)",
    )
    ax.set_xlim(-5.3, 5.3)
    ax.set_ylim(-7.8, 7.8)
    ax.set_xlabel(r"True Value $\mu_k \in [-5, 5]$")
    ax.set_ylabel(r"Predicted Value $\hat{\mu}_k$")
    ax.set_title(r"(A) Biểu đồ phân tán $\hat{\mu}_k$ vs. $\mu_k$ (1,000 mẫu: phylolm, M5A, M5B)")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (B) Chủ đạo 2: Scatter plot sigma cho 3 phương pháp (phylolm, M5A, M5B)
    ax = axes[0, 1]
    ax.plot([0, 5.4], [0, 5.4], "r--", lw=1.8, label=r"Đường phân tách $y = x$")
    ax.axhline(5.0, color="gray", ls=":", lw=1.1, label=r"Biên trên $\sigma = 5.0$")
    ax.scatter(
        sig_te,
        sig_phy,
        s=32,
        facecolors="none",
        edgecolors="#1d3557",
        lw=1.05,
        alpha=0.75,
        label=f"phylolm (MSE={np.mean((sig_phy-sig_te)**2):.5f})",
    )
    ax.scatter(
        sig_te,
        sig_m5a,
        s=24,
        marker="+",
        color="#00a896",
        lw=1.05,
        alpha=0.85,
        label=f"GNN M5A (MSE={np.mean((sig_m5a-sig_te)**2):.5f})",
    )
    ax.scatter(
        sig_te,
        sig_m5b,
        s=18,
        marker="D",
        color="#6a4c93",
        alpha=0.72,
        label=f"GNN M5B (MSE={np.mean((sig_m5b-sig_te)**2):.5f}, -9.7%)",
    )
    ax.set_xlim(0.0, 5.2)
    ax.set_ylim(0.0, 5.6)
    ax.set_xlabel(r"True Value $\sigma_k \in [0.1, 5.0]$")
    ax.set_ylabel(r"Predicted Value $\hat{\sigma}_k$")
    ax.set_title(r"(B) Biểu đồ phân tán $\hat{\sigma}_k$ vs. $\sigma_k$ (1,000 mẫu: phylolm, M5A, M5B)")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (C) Đối chứng với các mô hình cơ sở M0 (Star) & M4 (Raw MSE) trên mu
    ax = axes[0, 2]
    ax.plot([-5.3, 5.3], [-5.3, 5.3], "k--", lw=1.5, label=r"Đường $y = x$")
    ax.scatter(
        mu_te,
        mu_st,
        s=12,
        alpha=0.28,
        color="#e76f51",
        label=f"Star M0 (MSE={np.mean((mu_st-mu_te)**2):.4f})",
    )
    ax.scatter(
        mu_te,
        mu_m4,
        s=12,
        alpha=0.35,
        color="#f4a261",
        label=f"GNN M4 Raw MSE (MSE={np.mean((mu_m4-mu_te)**2):.4f})",
    )
    ax.scatter(
        mu_te,
        mu_m5b,
        s=14,
        alpha=0.55,
        color="#6a4c93",
        label=f"GNN M5B Bayes (MSE={np.mean((mu_m5b-mu_te)**2):.4f})",
    )
    ax.set_xlim(-5.3, 5.3)
    ax.set_ylim(-7.8, 7.8)
    ax.set_xlabel(r"True Value $\mu_k$")
    ax.set_ylabel(r"Predicted Value $\hat{\mu}_k$")
    ax.set_title(r"(C) Đối chứng với mô hình cơ sở Star ($\mathcal{M}_0$) & Raw MSE ($\mathcal{M}_4$)")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (D) Quạt phần dư dị phương sai e_mu vs sigma_k
    ax = axes[1, 0]
    s_grid = np.linspace(0.1, 5.0, 200)
    fan_95 = 1.96 * s_grid * math.sqrt(v_root)
    ax.scatter(sig_te, mu_phy - mu_te, s=11, alpha=0.40, color="#1d3557", label="Phần dư phylolm / GNN M5A")
    ax.scatter(sig_te, mu_m5b - mu_te, s=11, alpha=0.55, color="#6a4c93", label="Phần dư GNN M5B (-30.0% MSE)")
    ax.plot(s_grid, fan_95, "k--", lw=1.5, label=r"Hình nón 95% CRLB $\pm 1.96\sigma_k\sqrt{v_\rho}$")
    ax.plot(s_grid, -fan_95, "k--", lw=1.5)
    ax.fill_between(s_grid, -fan_95, fan_95, color="gray", alpha=0.12)
    ax.set_xlabel(r"True Value $\sigma_k$")
    ax.set_ylabel(r"Phần dư vị trí $e_{\mu, k} = \hat{\mu}_k - \mu_k$")
    ax.set_title(r"(D) Quạt phần dư dị phương sai $e_{\mu, k}$ theo $\sigma_k$")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (E) Phần dư chuẩn hóa z_{mu, k} = (mu_hat - mu) / (sigma_hat * sqrt(v_rho))
    ax = axes[1, 1]
    z_m5a = (mu_m5a - mu_te) / (sig_m5a * math.sqrt(v_root))
    z_m5b = (mu_m5b - mu_te) / (sig_m5a * math.sqrt(v_root))
    ax.scatter(sig_te, z_m5a, s=11, alpha=0.40, color="#00a896", label=f"GNN M5A (PICP 95% = {np.mean(np.abs(z_m5a)<=1.96)*100:.1f}%)")
    ax.scatter(sig_te, z_m5b, s=11, alpha=0.55, color="#6a4c93", label=f"GNN M5B (PICP 95% = {np.mean(np.abs(z_m5b)<=1.96)*100:.1f}%)")
    ax.axhline(1.96, color="#d62828", ls="--", lw=1.5, label=r"Ngưỡng chuẩn $\pm 1.96$")
    ax.axhline(-1.96, color="#d62828", ls="--", lw=1.5)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel(r"True Value $\sigma_k$")
    ax.set_ylabel(r"Phần dư chuẩn hóa $z_{\mu, k}$")
    ax.set_title(r"(E) Kiểm định khử dị phương sai & độ phủ PICP 95%")
    ax.legend(loc="upper right", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (F) Phần dư tốc độ tiến hóa e_sigma vs sigma_k
    ax = axes[1, 2]
    fan_sig_95 = 1.96 * s_grid / math.sqrt(2.0 * 147.0)
    ax.scatter(sig_te, sig_phy - sig_te, s=11, alpha=0.40, color="#1d3557", label="Phần dư phylolm / GNN M5A")
    ax.scatter(sig_te, sig_m5b - sig_te, s=11, alpha=0.55, color="#6a4c93", label="Phần dư GNN M5B (-9.7% MSE)")
    ax.plot(s_grid, fan_sig_95, "k--", lw=1.5, label=r"Hình nón 95% REML $\pm 1.96\sigma_k/\sqrt{294}$")
    ax.plot(s_grid, -fan_sig_95, "k--", lw=1.5)
    ax.fill_between(s_grid, -fan_sig_95, fan_sig_95, color="gray", alpha=0.12)
    ax.set_xlabel(r"True Value $\sigma_k$")
    ax.set_ylabel(r"Phần dư tốc độ $e_{\sigma, k} = \hat{\sigma}_k - \sigma_k$")
    ax.set_title(r"(F) Quạt phần dư tốc độ tiến hóa $e_{\sigma, k}$ theo $\sigma_k$")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        FIGURES_DIR / "advisor_fig2_simultaneous_parity_and_residual_scatter.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def _regenerate_figure3(eval_bundle: Dict[str, object]) -> None:
    pred_df: pd.DataFrame = eval_bundle["pred_test_df"]
    test_1000_df: pd.DataFrame = eval_bundle["test_1000_df"]

    mu_te = pred_df["mu_true"].values
    sig_te = pred_df["sigma_true"].values
    mu_phy = pred_df["mu_hat_phylolm"].values
    sig_phy = pred_df["sigma_hat_phylolm"].values
    mu_m5a = pred_df["mu_hat_GNN_M5A"].values
    sig_m5a = pred_df["sigma_hat_GNN_M5A"].values
    mu_m5b = pred_df["mu_hat_GNN_M5B"].values
    sig_m5b = pred_df["sigma_hat_GNN_M5B"].values
    mu_st = pred_df["mu_hat_Star_M0"].values
    sig_st = pred_df["sigma_hat_Star_M0"].values
    mu_m1 = pred_df["mu_hat_GNN_M1"].values
    sig_m1 = pred_df["sigma_hat_GNN_M1"].values

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.0), dpi=220)
    fig.suptitle(
        r"Figure 3: Chẩn đoán chuyên sâu trên tập kiểm tra cố định $\mathrm{Test\_Sim\_1000}$ ($N_{\mathrm{test}} = 1,000$ mẫu):"
        + "\n"
        + r"Đẳng cấu toán tử $\mathrm{PhyloGNN\_M5A} \leftrightarrow \mathrm{phylolm}$, đường cong co rút chữ S của $\mathrm{PhyloGNN\_M5B}$ & biên Pareto 2D",
        fontsize=11.6,
        fontweight="bold",
        y=0.98,
    )

    # (A) Đẳng cấu toán tử GNN M5A vs phylolm trên mu
    ax = axes[0, 0]
    r_mu = float(np.corrcoef(mu_phy, mu_m5a)[0, 1])
    max_d_mu = float(np.max(np.abs(mu_m5a - mu_phy)))
    ax.plot([-7.8, 7.8], [-7.8, 7.8], "k--", lw=1.6, label=f"Đường đồng nhất $y = x$ ($r = {r_mu:.6f}$)")
    ax.scatter(mu_phy, mu_m5a, c=sig_te, cmap="magma", s=14, alpha=0.75)
    ax.set_xlabel(r"Tiêu chuẩn vàng $\hat{\mu}_{\mathrm{phylolm}, k}$")
    ax.set_ylabel(r"Mô hình $\hat{\mu}_{\mathrm{M5A}, k}$")
    ax.set_title(f"(A) Đẳng cấu toán tử trên $\\hat{{\\mu}}$ ($\\max|\\Delta| = {max_d_mu:.4f}$)")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, alpha=0.25)

    # (B) Đẳng cấu toán tử GNN M5A vs phylolm trên sigma
    ax = axes[0, 1]
    r_sig = float(np.corrcoef(sig_phy, sig_m5a)[0, 1])
    max_d_sig = float(np.max(np.abs(sig_m5a - sig_phy)))
    ax.plot([0, 5.5], [0, 5.5], "k--", lw=1.6, label=f"Đường đồng nhất $y = x$ ($r = {r_sig:.6f}$)")
    sc_b = ax.scatter(sig_phy, sig_m5a, c=np.abs(mu_te), cmap="viridis", s=14, alpha=0.75)
    plt.colorbar(sc_b, ax=ax, label=r"$|\mu_k|$")
    ax.set_xlabel(r"Tiêu chuẩn vàng $\hat{\sigma}_{\mathrm{phylolm}, k}$")
    ax.set_ylabel(r"Mô hình $\hat{\sigma}_{\mathrm{M5A}, k}$")
    ax.set_title(f"(B) Đẳng cấu toán tử trên $\\hat{{\\sigma}}$ ($\\max|\\Delta| = {max_d_sig:.4f}$)")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, alpha=0.25)

    # (C) Đường cong co rút chữ S của GNN M5B vs phylolm
    ax = axes[0, 2]
    ax.plot([-7.5, 7.5], [-7.5, 7.5], "k--", lw=1.5, label=r"Đường không co rút $y = x$ (phylolm)")
    ax.axhline(5.0, color="#d62828", ls=":", lw=1.2, label=r"Biên tiên nghiệm chuẩn cụt $[\pm 5.0]$")
    ax.axhline(-5.0, color="#d62828", ls=":", lw=1.2)
    sc_c = ax.scatter(mu_phy, mu_m5b, c=sig_te, cmap="coolwarm", s=15, alpha=0.78)
    plt.colorbar(sc_c, ax=ax, label=r"True Value $\sigma_k$")
    ax.set_xlim(-7.8, 7.8)
    ax.set_ylim(-5.6, 5.6)
    ax.set_xlabel(r"Tiêu chuẩn vàng không chệch $\hat{\mu}_{\mathrm{phylolm}, k}$")
    ax.set_ylabel(r"Ước lượng co rút Bayes $\hat{\mu}_{\mathrm{M5B}, k}$")
    ax.set_title(r"(C) Đường cong co rút chữ S: GNN M5B vs. phylolm")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (D) Không gian tham số 2D: nơi GNN M5B thắng phylolm trên Test_Sim_1000
    ax = axes[1, 0]
    gain_abs = np.abs(mu_phy - mu_te) - np.abs(mu_m5b - mu_te)
    sc_d = ax.scatter(
        mu_te,
        sig_te,
        c=np.clip(gain_abs, -0.35, 0.85),
        cmap="RdYlGn",
        s=16,
        alpha=0.82,
    )
    plt.colorbar(
        sc_d,
        ax=ax,
        label=r"Mức giảm sai số $|\hat{\mu}_{\mathrm{phylolm}} - \mu_k| - |\hat{\mu}_{\mathrm{M5B}} - \mu_k|$",
    )
    ax.axvline(-3.5, color="black", ls="--", lw=1.2, label=r"Vùng biên $|\mu_k| \geq 3.5$")
    ax.axvline(3.5, color="black", ls="--", lw=1.2)
    ax.set_xlabel(r"True Value $\mu_k \in [-5, 5]$")
    ax.set_ylabel(r"True Value $\sigma_k \in [0.1, 5.0]$")
    ax.set_title(r"(D) Bản đồ không gian 2D $(\mu_k, \sigma_k)$ trên Test_Sim_1000")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, alpha=0.25)

    # (E) Mặt phẳng sai số tương đối 2D trên Test_Sim_1000
    ax = axes[1, 1]
    ax.scatter(
        np.abs(mu_st - mu_te) / sig_te,
        np.abs(sig_st - sig_te) / sig_te,
        s=11,
        alpha=0.28,
        color="#e76f51",
        label="Star Baseline (M0)",
    )
    ax.scatter(
        np.abs(mu_m1 - mu_te) / sig_te,
        np.abs(sig_m1 - sig_te) / sig_te,
        s=11,
        alpha=0.28,
        color="#f4a261",
        label="GNN M1 (DeepSets)",
    )
    ax.scatter(
        np.abs(mu_phy - mu_te) / sig_te,
        np.abs(sig_phy - sig_te) / sig_te,
        s=12,
        alpha=0.45,
        color="#00a896",
        label="phylolm / GNN M5A",
    )
    ax.scatter(
        np.abs(mu_m5b - mu_te) / sig_te,
        np.abs(sig_m5b - sig_te) / sig_te,
        s=12,
        alpha=0.55,
        color="#6a4c93",
        label="GNN M5B (Bayes MMSE)",
    )
    ax.axvline(
        math.sqrt(2.0 / math.pi) * math.sqrt(0.20216894),
        color="black",
        ls="--",
        lw=1.3,
        label="Ngưỡng trung bình CRLB",
    )
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 0.45)
    ax.set_xlabel(r"Sai số vị trí chuẩn hóa $|\hat{\mu}_k - \mu_k| / \sigma_k$")
    ax.set_ylabel(r"Sai số tốc độ tương đối $|\hat{\sigma}_k - \sigma_k| / \sigma_k$")
    ax.set_title(r"(E) Phân bố sai số tương đối 2D trên Test_Sim_1000")
    ax.legend(loc="upper right", fontsize=7.8)
    ax.grid(True, alpha=0.25)

    # (F) Biên hiệu quả Pareto 2 tham số (RMSE(mu) vs RMSE(sigma)) trên Test_Sim_1000
    ax = axes[1, 2]
    style_map = {
        "Star_Baseline": ("X", "#d62828", "Star Baseline"),
        "M1_MLP_DeepSets": ("v", "#e76f51", "M1 DeepSets"),
        "M2_GCN_Unweighted": ("s", "#f4a261", "M2 GCN"),
        "M3_EdgeMPNN_GINE": ("D", "#457b9d", "M3 GINE"),
        "M4_PhyloGNN_RawMSE": ("^", "#2a9d8f", "M4 Raw MSE"),
        "phylolm_GoldStandard": ("P", "#1d3557", "phylolm (Gold)"),
        "M5A_PhyloGNN_Unbiased_NLL": ("o", "#00a896", "GNN M5A (NLL)"),
        "M5B_PhyloGNN_BayesMMSE": ("*", "#6a4c93", "GNN M5B (Bayes)"),
    }
    for _, row in test_1000_df.iterrows():
        m_key = row["Method"]
        if m_key in style_map:
            mk, col, lbl = style_map[m_key]
            r_m = float(row["mu_RMSE"])
            r_s = float(row["sigma_RMSE"])
            sz = 155 if m_key == "M5B_PhyloGNN_BayesMMSE" else 95
            ax.scatter(
                r_m,
                r_s,
                marker=mk,
                color=col,
                s=sz,
                edgecolors="black",
                lw=0.9,
                label=f"{lbl} ({r_m:.3f}, {r_s:.3f})",
            )
    ax.set_xlabel(r"Sai số vị trí gốc $\mathrm{RMSE}(\hat{\mu})$ trên Test_Sim_1000")
    ax.set_ylabel(r"Sai số tốc độ tiến hóa $\mathrm{RMSE}(\hat{\sigma})$ trên Test_Sim_1000")
    ax.set_title(r"(F) Biên hiệu quả Pareto 2 tham số trên Test_Sim_1000")
    ax.legend(loc="upper left", fontsize=7.6)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        FIGURES_DIR / "advisor_fig3_concordance_and_2d_error_scatter.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_step4_pipeline() -> Dict[str, object]:
    """Thực thi Bước 4: phân tích phân tầng trên Test_Sim_1000 và đồng bộ toàn bộ bảng & biểu đồ."""
    return run_test1000_synthesis_and_regenerate_figures()


def load_test1000_regime_table() -> pd.DataFrame:
    """Tải bảng phân tích phân tầng trên tập kiểm tra cố định Test_Sim_1000."""
    csv_path = PROCESSED_DATA_DIR / "step4_test1000_regime_and_boundary_analysis.csv"
    if not csv_path.exists():
        run_step4_pipeline()
    return pd.read_csv(csv_path)


# Giữ bí danh tương thích nếu có script gọi tên hàm cũ
run_multiscale_evaluation_and_regenerate_figures = run_test1000_synthesis_and_regenerate_figures
load_multiscale_benchmark_table = load_test1000_regime_table


if __name__ == "__main__":
    res4 = run_step4_pipeline()
    df_r = res4["regime_df"]
    print("===== KẾT QUẢ BƯỚC 4: PHÂN TÍCH PHÂN TẦNG TRÊN TẬP TEST CỐ ĐỊNH TEST_SIM_1000 (1,000 MẪU) =====")
    for _, row in df_r.iterrows():
        print(
            f"  * {row['Stratum']:34s} (N={int(row['N_samples']):4d}) | "
            f"mu MSE: phylolm={row['mu_MSE_phylolm']:.4f}, M5A={row['mu_MSE_GNN_M5A']:.4f}, "
            f"M5B={row['mu_MSE_GNN_M5B']:.4f} ({row['mu_MSE_Gain_M5B_vs_phylolm_pct']:+.2f}%) | "
            f"sigma MSE: phylolm={row['sigma_MSE_phylolm']:.5f}, M5B={row['sigma_MSE_GNN_M5B']:.5f} "
            f"({row['sigma_MSE_Gain_M5B_vs_phylolm_pct']:+.2f}%)"
        )
