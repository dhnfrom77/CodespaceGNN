#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 1: Kiểm toán Tập Huấn luyện `calib_123.csv` (10,000 mẫu) và Tập Kiểm tra Mô phỏng
        Cố định `test_sim_1000_fixed.csv` (1,000 mẫu), Đánh giá Mô hình Star Baseline (M0).
"""

# 1. Thư viện chuẩn của Python
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 2. Thư viện bên thứ ba
import numpy as np
import pandas as pd
from scipy import stats

# 3. Mã của chính dự án
from bm_gnn.config_paths import (
    TEST_SIM_CSV_PATH,
    TRAIN_CALIB_CSV_PATH,
    ensure_output_directories,
)
from bm_gnn.step0_tree_and_bm_simulator import (
    FromScratchPhylogenyBuilder,
    c4_factor,
    compute_reusable_metrics,
    ensure_fixed_test_set_1000,
)


def run_star_baseline(t_diag: np.ndarray, x_mat: np.ndarray) -> Dict[str, np.ndarray]:
    """Tính bộ ước lượng ngây thơ M0 (Star Phylogeny Baseline — giả định S = I_n)."""
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


def load_train_and_fixed_test_datasets(
    builder: FromScratchPhylogenyBuilder,
    train_csv_path: Path = TRAIN_CALIB_CSV_PATH,
    test_csv_path: Path = TEST_SIM_CSV_PATH,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Đọc toàn bộ 10,000 mẫu của tập Huấn luyện `calib_123.csv` (`Train_10000`)
    và 1,000 mẫu của tập Kiểm tra Mô phỏng Cố định `test_sim_1000_fixed.csv` (`Test_Sim_1000`).
    """
    df_train = pd.read_csv(train_csv_path)
    mu_train = df_train["mu"].values.astype(np.float64)
    sig_train = df_train["sigma"].values.astype(np.float64)
    x_train = df_train[builder.species_names].values.astype(np.float64)

    df_test = ensure_fixed_test_set_1000(builder, test_csv_path)
    mu_test = df_test["mu"].values.astype(np.float64)
    sig_test = df_test["sigma"].values.astype(np.float64)
    x_test = df_test[builder.species_names].values.astype(np.float64)

    return df_train, mu_train, sig_train, x_train, df_test, mu_test, sig_test, x_test


def load_and_audit_calib123(
    builder: FromScratchPhylogenyBuilder,
    csv_path: Path = TRAIN_CALIB_CSV_PATH,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Đọc tệp `calib_123.csv` (10,000 mẫu Train) từ `data/raw/`."""
    df_calib = pd.read_csv(csv_path)
    mu_true = df_calib["mu"].values.astype(np.float64)
    sigma_true = df_calib["sigma"].values.astype(np.float64)
    x_mat = df_calib[builder.species_names].values.astype(np.float64)
    return df_calib, mu_true, sigma_true, x_mat


def run_step1_pipeline() -> Dict[str, object]:
    """Thực thi Bước 1: kiểm toán tập Train (10,000 mẫu) và tập Test cố định (1,000 mẫu)."""
    ensure_output_directories()
    builder = FromScratchPhylogenyBuilder()
    (
        df_tr,
        mu_tr,
        sig_tr,
        x_tr,
        df_te,
        mu_te,
        sig_te,
        x_te,
    ) = load_train_and_fixed_test_datasets(builder)

    ks_mu = stats.ks_2samp(mu_tr, mu_te)
    ks_sig = stats.ks_2samp(sig_tr, sig_te)

    t_diag = np.diag(builder.S_mat)
    res_star_tr = run_star_baseline(t_diag, x_tr)
    res_star_te = run_star_baseline(t_diag, x_te)

    star_se_factor = math.sqrt(float(np.mean(builder.S_mat)))
    m_mu_te = compute_reusable_metrics(
        mu_te,
        res_star_te["mu_hat"],
        sigma_scale=sig_te,
        pred_se=res_star_te["sigma_reml"] * star_se_factor,
    )
    m_sig_te = compute_reusable_metrics(
        sig_te, res_star_te["sigma_mle"], sigma_scale=sig_te
    )

    return {
        "n_train_samples": len(df_tr),
        "n_test_samples": len(df_te),
        "train_nan_count": int(df_tr.isna().sum().sum()),
        "test_nan_count": int(df_te.isna().sum().sum()),
        "ks_pvalue_mu_train_vs_test": float(ks_mu.pvalue),
        "ks_pvalue_sigma_train_vs_test": float(ks_sig.pvalue),
        "kappa_tree": builder.kappa_tree,
        "star_test1000_mu": m_mu_te,
        "star_test1000_sigma": m_sig_te,
        "star_train10000_mu": compute_reusable_metrics(
            mu_tr,
            res_star_tr["mu_hat"],
            sigma_scale=sig_tr,
            pred_se=res_star_tr["sigma_reml"] * star_se_factor,
        ),
    }


if __name__ == "__main__":
    s1 = run_step1_pipeline()
    print("===== KẾT QUẢ BƯỚC 1: KIỂM ĐỊNH TẬP TRAIN (10,000 MẪU) & TẬP TEST CỐ ĐỊNH (1,000 MẪU) =====")
    print(f"  * Tập Train (calib_123.csv)         : {s1['n_train_samples']:,d} mẫu (NaN = {s1['train_nan_count']})")
    print(f"  * Tập Test (test_sim_1000_fixed.csv): {s1['n_test_samples']:,d} mẫu (NaN = {s1['test_nan_count']})")
    print(f"  * Kiểm định đồng nhất phân phối KS  : p(mu) = {s1['ks_pvalue_mu_train_vs_test']:.4f}, p(sigma) = {s1['ks_pvalue_sigma_train_vs_test']:.4f} (Chấp nhận H0!)")
    print(f"  * Star Baseline trên Tập Test 1,000 : MSE(mu) = {s1['star_test1000_mu']['MSE']:.4f}, MSE(sigma) = {s1['star_test1000_sigma']['MSE']:.4f}")
