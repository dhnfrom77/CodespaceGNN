#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 2: Thuật toán Cắt tỉa Cấu trúc 3 Điểm Tuyến tính O(n) của `phylolm` (Ho & Ané, 2014),
        Kiểm chứng Tương đương Đại số (< 10^-14) với Cholesky GLS & Felsenstein PIC trên
        cả Tập Train (10,000 mẫu) và Tập Test Mô phỏng Cố định (1,000 mẫu).
"""

# 1. Thư viện chuẩn của Python
import math
import sys
from pathlib import Path
from typing import Dict

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 2. Thư viện bên thứ ba
import numpy as np
from scipy import linalg

# 3. Mã của chính dự án
from bm_gnn.step0_tree_and_bm_simulator import (
    FromScratchPhylogenyBuilder,
    c4_factor,
    compute_reusable_metrics,
)
from bm_gnn.step1_data_diagnostics_and_star import load_train_and_fixed_test_datasets


def run_phylolm_3pt_pruning_from_scratch(
    builder: FromScratchPhylogenyBuilder,
    x_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Thuật toán Cắt tỉa Cấu trúc 3 Điểm Tuyến tính O(n) của Ho & Ané (2014, `phylolm`)."""
    m_samples, n_leaves = x_mat.shape
    df = n_leaves - 1
    c4_val = c4_factor(df)

    logd, u1, u_x, vec11, p1, pp = {}, {}, {}, {}, {}, {}
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
    """Tính nghiệm bằng Cholesky GLS O(n^3) và Quy hoạch Động Felsenstein PIC O(n)."""
    m_samples, n_leaves = x_mat.shape
    df = n_leaves - 1
    c4_val = c4_factor(df)

    mu_chol = x_mat @ builder.w_gls
    residuals = (x_mat - mu_chol[:, None]).T
    whitened = linalg.solve_triangular(builder.L_S, residuals, lower=True)
    q_chol = np.sum(whitened ** 2, axis=0)
    sigma_reml_chol = np.sqrt(q_chol / df) / c4_val

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
        "H_nodes": h_nodes,
        "v_nodes": v_nodes,
    }


def run_step2_pipeline() -> Dict[str, object]:
    """Thực thi Bước 2 trên cả tập Train (10,000 mẫu) và tập Test cố định (1,000 mẫu)."""
    builder = FromScratchPhylogenyBuilder()
    (
        _,
        mu_tr,
        sig_tr,
        x_tr,
        _,
        mu_te,
        sig_te,
        x_te,
    ) = load_train_and_fixed_test_datasets(builder)

    res_phylo_tr = run_phylolm_3pt_pruning_from_scratch(builder, x_tr)
    res_phylo_te = run_phylolm_3pt_pruning_from_scratch(builder, x_te)
    res_other_te = run_cholesky_gls_and_pic_solvers(builder, x_te)

    max_diff_mu_pic = float(np.max(np.abs(res_phylo_te["mu_hat"] - res_other_te["mu_pic"])))
    max_diff_mu_chol = float(np.max(np.abs(res_phylo_te["mu_hat"] - res_other_te["mu_chol"])))

    t_reml_tr = res_phylo_tr["Q_internal"] / (147.0 * (sig_tr ** 2))
    t_reml_te = res_phylo_te["Q_internal"] / (147.0 * (sig_te ** 2))

    se_te = res_phylo_te["sigma_reml"] * math.sqrt(builder.v_root)
    metrics_mu_te = compute_reusable_metrics(
        mu_te, res_phylo_te["mu_hat"], sigma_scale=sig_te, pred_se=se_te
    )
    metrics_sig_te = compute_reusable_metrics(
        sig_te, res_phylo_te["sigma_reml"], sigma_scale=sig_te
    )

    return {
        "max_diff_mu_phylolm_vs_pic": max_diff_mu_pic,
        "max_diff_mu_phylolm_vs_cholesky": max_diff_mu_chol,
        "t_reml_train_mean": float(np.mean(t_reml_tr)),
        "t_reml_train_se": float(np.std(t_reml_tr, ddof=1) / math.sqrt(len(t_reml_tr))),
        "t_reml_test_mean": float(np.mean(t_reml_te)),
        "t_reml_test_se": float(np.std(t_reml_te, ddof=1) / math.sqrt(len(t_reml_te))),
        "phylolm_test1000_mu": metrics_mu_te,
        "phylolm_test1000_sigma": metrics_sig_te,
    }


if __name__ == "__main__":
    info2 = run_step2_pipeline()
    print("===== KẾT QUẢ BƯỚC 2: TIÊU CHUẨN VÀNG phylolm TRÊN TẬP TRAIN (10,000) & TEST CỐ ĐỊNH (1,000) =====")
    print(f"  * Sai lệch cực đại |mu_phylolm - mu_PIC|      : {info2['max_diff_mu_phylolm_vs_pic']:.3e}")
    print(f"  * Sai lệch cực đại |mu_phylolm - mu_Cholesky| : {info2['max_diff_mu_phylolm_vs_cholesky']:.3e}")
    print(f"  * Kiểm định t_hat_REML (Train 10,000 mẫu)     : {info2['t_reml_train_mean']:.6f} ± {info2['t_reml_train_se']:.6f}")
    print(f"  * Kiểm định t_hat_REML (Test cố định 1,000)   : {info2['t_reml_test_mean']:.6f} ± {info2['t_reml_test_se']:.6f}")
    m_mu = info2["phylolm_test1000_mu"]
    m_sg = info2["phylolm_test1000_sigma"]
    print(f"  * phylolm (Test 1,000) trên mu    : MSE = {m_mu['MSE']:.4f}, RMSE = {m_mu['RMSE']:.4f}, NRMSE_sig = {m_mu['NRMSE_by_sigma']:.4f}, PICP_95 = {m_mu['PICP_95_pct']:.1f}%")
    print(f"  * phylolm (Test 1,000) trên sigma : MSE = {m_sg['MSE']:.5f}, RMSE = {m_sg['RMSE']:.4f}, MAPE = {m_sg['MAPE_pct']:.2f}%, R2 = {m_sg['R2']:.4f}")
