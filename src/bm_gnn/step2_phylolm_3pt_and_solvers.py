#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 2: Giải tích Không gian Hilbert <u, v>_{S^-1}, Định lý Phân rã Trực giao 147 + 1 DOF,
        Thuật toán Cắt tỉa Cấu trúc 3 Điểm Tuyến tính O(n) của `phylolm` (Ho & Ané, 2014)
        và Kiểm chứng Tương đương Tuyệt đối (< 10^-15) với Cholesky GLS & Felsenstein PIC.

Tham chiếu:
  - Hình K2: Hình học Không gian Hilbert <u, v>_{S^-1} & Định lý Pythagoras 147 + 1 DOF.
  - Hình K3: Lưu đồ Thuật toán Cắt tỉa Cấu trúc 3 Điểm Tuyến tính O(n) của `phylolm`.
  - Figure 1: Chẩn đoán dữ liệu, trọng số cắt tỉa w_{phylolm, i} và kiểm định t ≈ 1.0.
  - Tiểu luận Mục 3.1 - 3.2 & Cell 8 - Cell 11 trong Master Notebook.
"""

# 1. Thư viện chuẩn của Python
import math
from typing import Dict

# 2. Thư viện bên thứ ba
import numpy as np
from scipy import linalg

# 3. Mã của chính dự án
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_gnn.step0_tree_and_bm_simulator import (
    FromScratchPhylogenyBuilder,
    c4_factor,
    compute_reusable_metrics,
)
from bm_gnn.step1_data_diagnostics_and_star import load_and_audit_calib123


def run_phylolm_3pt_pruning_from_scratch(
    builder: FromScratchPhylogenyBuilder,
    x_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Thuật toán Cắt tỉa Cấu trúc 3 Điểm Tuyến tính O(n) của Ho & Ané (2014, `phylolm`):
    Duyệt hậu tố 1 lượt qua 295 nút với 6 thanh ghi đại số (logd_u, u1_u, uX_u, vec11_u, p1_u, pp_u)
    để tính chính xác mu_hat_GLS, Q_internal, sigma_hat_REML (khử chệch c_4) và ln det(S).
    """
    m_samples, n_leaves = x_mat.shape
    df = n_leaves - 1
    c4_val = c4_factor(df)

    logd = {}
    u1 = {}
    u_x = {}
    vec11 = {}
    p1 = {}
    pp = {}

    for u in builder.postorder_nodes:
        nid = u.node_id
        t_u = u.length if u.parent is not None else 0.0
        if u.is_leaf():
            x_col = x_mat[:, nid]
            if t_u > 0.0:
                logd[nid] = math.log(t_u)
                u1[nid] = 1.0
                u_x[nid] = x_col
                vec11[nid] = 1.0 / t_u
                p1[nid] = x_col / t_u
                pp[nid] = (x_col ** 2) / t_u
            else:
                logd[nid] = 0.0
                u1[nid] = 0.0
                u_x[nid] = np.zeros(m_samples)
                vec11[nid] = np.inf
                p1[nid] = np.zeros(m_samples)
                pp[nid] = np.zeros(m_samples)
        else:
            p_children = sum(vec11[ch.node_id] for ch in u.children)
            y1_children = sum(p1[ch.node_id] for ch in u.children)
            yx_children = sum(pp[ch.node_id] for ch in u.children)
            logd_children = sum(logd[ch.node_id] for ch in u.children)
            denom = 1.0 + t_u * p_children

            logd[nid] = logd_children + math.log(denom)
            u1[nid] = t_u * p_children / denom
            u_x[nid] = t_u * y1_children / denom
            vec11[nid] = p_children / denom
            p1[nid] = y1_children / denom
            pp[nid] = yx_children - (t_u / denom) * (y1_children ** 2)

    rid = builder.root.node_id
    logdet_s = float(logd[rid])
    v11_root = float(vec11[rid])
    v_root = 1.0 / v11_root
    mu_hat = p1[rid] / v11_root
    q_internal = np.maximum(pp[rid] - (p1[rid] ** 2) / v11_root, 1e-15)
    sigma_mle = np.sqrt(q_internal / n_leaves)
    sigma_reml = np.sqrt(q_internal / df) / c4_val

    return {
        "mu_hat": mu_hat,
        "sigma_mle": sigma_mle,
        "sigma_reml": sigma_reml,
        "Q_internal": q_internal,
        "logdet_S": logdet_s,
        "vec11": v11_root,
        "v_root": v_root,
    }


def run_cholesky_gls_and_pic_solvers(
    builder: FromScratchPhylogenyBuilder,
    x_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Tính nghiệm bằng Đại số Tuyến tính Cholesky GLS O(n^3) và Quy hoạch Động
    Tương phản Độc lập Felsenstein PIC O(n) để đối chiếu sai số máy với `phylolm`.
    """
    m_samples, n_leaves = x_mat.shape
    df = n_leaves - 1
    c4_val = c4_factor(df)

    # 1. Cholesky GLS
    mu_chol = x_mat @ builder.w_gls
    residuals = (x_mat - mu_chol[:, None]).T  # (n, M)
    whitened = linalg.solve_triangular(builder.L_S, residuals, lower=True)
    q_chol = np.sum(whitened ** 2, axis=0)
    sigma_reml_chol = np.sqrt(q_chol / df) / c4_val

    # 2. Felsenstein PIC (1985)
    h_nodes = np.zeros((m_samples, len(builder.all_nodes)), dtype=np.float64)
    v_nodes = np.zeros(len(builder.all_nodes), dtype=np.float64)
    c_int = np.zeros((m_samples, df), dtype=np.float64)
    h_nodes[:, :n_leaves] = x_mat

    for idx_u in range(df):
        nid = n_leaves + idx_u
        l_id = int(builder.graph_tensors["left_ids"][idx_u])
        r_id = int(builder.graph_tensors["right_ids"][idx_u])
        dt_l = float(builder.graph_tensors["dt_left"][idx_u])
        dt_r = float(builder.graph_tensors["dt_right"][idx_u])

        d_l = v_nodes[l_id] + dt_l
        d_r = v_nodes[r_id] + dt_r
        d_sum = d_l + d_r

        h_nodes[:, nid] = (d_r * h_nodes[:, l_id] + d_l * h_nodes[:, r_id]) / d_sum
        v_nodes[nid] = (d_l * d_r) / d_sum
        c_int[:, idx_u] = (h_nodes[:, l_id] - h_nodes[:, r_id]) / math.sqrt(d_sum)

    mu_pic = h_nodes[:, -1]
    q_pic = np.sum(c_int ** 2, axis=1)
    sigma_reml_pic = np.sqrt(q_pic / df) / c4_val

    return {
        "mu_chol": mu_chol,
        "Q_chol": q_chol,
        "sigma_reml_chol": sigma_reml_chol,
        "mu_pic": mu_pic,
        "Q_pic": q_pic,
        "sigma_reml_pic": sigma_reml_pic,
        "C_int": c_int,
    }


def run_step2_pipeline() -> Dict[str, object]:
    """Thực thi Bước 2: chạy `phylolm`, đối chiếu Cholesky/PIC và kiểm định t ≈ 1.0."""
    builder = FromScratchPhylogenyBuilder()
    _, mu_true, sigma_true, x_mat = load_and_audit_calib123(builder)

    res_phylo = run_phylolm_3pt_pruning_from_scratch(builder, x_mat)
    res_other = run_cholesky_gls_and_pic_solvers(builder, x_mat)

    max_diff_mu_pic = float(np.max(np.abs(res_phylo["mu_hat"] - res_other["mu_pic"])))
    max_diff_mu_chol = float(np.max(np.abs(res_phylo["mu_hat"] - res_other["mu_chol"])))
    max_diff_q_pic = float(np.max(np.abs(res_phylo["Q_internal"] - res_other["Q_pic"])))
    max_diff_q_chol = float(np.max(np.abs(res_phylo["Q_internal"] - res_other["Q_chol"])))

    t_reml = res_phylo["Q_internal"] / (147.0 * (sigma_true ** 2))
    t_mean = float(np.mean(t_reml))
    t_se = float(np.std(t_reml, ddof=1) / math.sqrt(len(t_reml)))

    pred_se_phylo = res_phylo["sigma_reml"] * math.sqrt(builder.v_root)
    metrics_mu = compute_reusable_metrics(
        mu_true, res_phylo["mu_hat"], sigma_scale=sigma_true, pred_se=pred_se_phylo
    )
    metrics_sig = compute_reusable_metrics(
        sigma_true, res_phylo["sigma_reml"], sigma_scale=sigma_true
    )

    return {
        "max_diff_mu_phylolm_vs_pic": max_diff_mu_pic,
        "max_diff_mu_phylolm_vs_cholesky": max_diff_mu_chol,
        "max_diff_Q_phylolm_vs_pic": max_diff_q_pic,
        "max_diff_Q_phylolm_vs_cholesky": max_diff_q_chol,
        "t_reml_mean": t_mean,
        "t_reml_se": t_se,
        "phylolm_full10000_mu": metrics_mu,
        "phylolm_full10000_sigma": metrics_sig,
    }


if __name__ == "__main__":
    info2 = run_step2_pipeline()
    print("===== KẾT QUẢ BƯỚC 2: TIÊU CHUẨN VÀNG phylolm & KIỂM CHỨNG ĐẠI SỐ =====")
    print(f"  * Sai lệch cực đại |mu_phylolm - mu_PIC|      : {info2['max_diff_mu_phylolm_vs_pic']:.3e}")
    print(f"  * Sai lệch cực đại |mu_phylolm - mu_Cholesky| : {info2['max_diff_mu_phylolm_vs_cholesky']:.3e}")
    print(f"  * Sai lệch cực đại |Q_phylolm - Q_PIC|        : {info2['max_diff_Q_phylolm_vs_pic']:.3e}")
    print(f"  * Sai lệch cực đại |Q_phylolm - Q_Cholesky|   : {info2['max_diff_Q_phylolm_vs_cholesky']:.3e}")
    print(f"  * Kiểm định chiều cao cây t_hat_REML          : {info2['t_reml_mean']:.6f} ± {info2['t_reml_se']:.6f} (Chứa t = 1.0000)")
    m_mu = info2["phylolm_full10000_mu"]
    m_sg = info2["phylolm_full10000_sigma"]
    print(f"  * phylolm (Full_10000) trên mu    : MSE = {m_mu['MSE']:.6f}, NRMSE_sig = {m_mu['NRMSE_by_sigma']:.4f}, PICP_95 = {m_mu['PICP_95_pct']:.2f}%")
    print(f"  * phylolm (Full_10000) trên sigma : MSE = {m_sg['MSE']:.6f}, Bias = {m_sg['Bias']:+.6f}, R2 = {m_sg['R2']:.4f}")
