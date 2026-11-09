#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kịch bản Điều phối Tổng thể (Master CLI Runner) cho `CodespaceGNN`:
Chạy tuần tự toàn bộ quy trình Học máy từ Bước 0 -> Bước 4:
  - Tập Train: 10,000 mẫu trong `data/raw/calib_123.csv`
  - Tập Test cố định: 1,000 mẫu mô phỏng trong `data/raw/test_sim_1000_fixed.csv`
  - Biểu đồ chủ đạo: `figures/flagship_test1000_pred_vs_true_mu_sigma_scatter.png`
"""

# 1. Thư viện chuẩn của Python
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 3. Mã của chính dự án
from bm_gnn.config_paths import FLAGSHIP_SCATTER_PATH, TEST_SIM_CSV_PATH, TRAIN_CALIB_CSV_PATH
from bm_gnn.step0_tree_and_bm_simulator import run_step0_verification
from bm_gnn.step1_data_diagnostics_and_star import run_step1_pipeline
from bm_gnn.step2_phylolm_3pt_and_solvers import run_step2_pipeline
from bm_gnn.step3_gnn_architectures import verify_equivariance_and_invariance_properties
from bm_gnn.step4_multiscale_and_synthesis import run_step4_pipeline


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 92)
    print("  KHỞI CHẠY PIPELINE HỌC MÁY BM-GNN TRONG CODESPACEGNN (TRAIN = 10,000 | TEST CỐ ĐỊNH = 1,000)")
    print("=" * 92)

    s0 = run_step0_verification()
    print(
        f"\n[Bước 0 Hoàn tất] Đồ thị |V|={s0['n_total_nodes']} nút, |E_dir|={s0['n_directed_edges']} cạnh | "
        f"ln det(S)={s0['logdet_S']:.6f} | vec11={s0['vec11']:.6f} | v_root={s0['v_root']:.6f}"
    )

    s1 = run_step1_pipeline()
    print(
        f"[Bước 1 Hoàn tất] Tập Train ({TRAIN_CALIB_CSV_PATH.name}): {s1['n_train_samples']:,d} mẫu | "
        f"Tập Test cố định ({TEST_SIM_CSV_PATH.name}): {s1['n_test_samples']:,d} mẫu | "
        f"KS p-value(mu)={s1['ks_pvalue_mu_train_vs_test']:.4f}, p-value(sigma)={s1['ks_pvalue_sigma_train_vs_test']:.4f}"
    )

    s2 = run_step2_pipeline()
    print(
        f"[Bước 2 Hoàn tất] Đối chiếu phylolm vs. PIC/Cholesky sai số < {s2['max_diff_mu_phylolm_vs_cholesky']:.2e} | "
        f"t_REML(Train)={s2['t_reml_train_mean']:.6f}, t_REML(Test)={s2['t_reml_test_mean']:.6f}"
    )

    s3a = verify_equivariance_and_invariance_properties()
    print(
        f"[Bước 3A Hoàn tất] Kiểm chứng Đẳng biến/Bất biến dịch chuyển: "
        f"sai số mu={s3a['translation_equivariance_max_error_mu']:.2e}, "
        f"sai số sigma={s3a['translation_invariance_max_error_sigma']:.2e}"
    )

    s4 = run_step4_pipeline()
    eval_b = s4["eval_bundle"]
    df_te = eval_b["test_1000_df"]
    print("\n[Bước 3B Hoàn tất] Kết quả trên Tập Test Mô phỏng Cố định (1,000 mẫu, huấn luyện trên 10,000 mẫu calib_123.csv):")
    for _, r in df_te.iterrows():
        print(
            f"  - {r['Method']:26s}: MSE(mu)={r['mu_MSE']:.4f}, RMSE(mu)={r['mu_RMSE']:.4f}, "
            f"NRMSE_sig={r['mu_NRMSE_sigma']:.4f}, R2(mu)={r['mu_R2']:.4f}, PICP95={r['mu_PICP_95_pct']:.1f}% | "
            f"MSE(sigma)={r['sigma_MSE']:.5f}, RMSE(sigma)={r['sigma_RMSE']:.4f}, R2(sigma)={r['sigma_R2']:.4f}"
        )

    print(f"\n[Biểu đồ Chủ đạo] Đã xuất 2-Panel Scatter Plot (1,000 mẫu x 3 phương pháp) tại:\n  -> {FLAGSHIP_SCATTER_PATH}")

    elapsed = time.perf_counter() - t_start
    print("=" * 92)
    print(f"  HOÀN TẤT TOÀN BỘ PIPELINE CODESPACEGNN TRONG {elapsed:.2f} GIÂY!")
    print("=" * 92)


if __name__ == "__main__":
    main()
