#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kịch bản Điều phối Tổng thể (Master CLI Runner):
Chạy tuần tự toàn bộ 5 bước của chương trình nghiên cứu (Bước 0 -> Bước 4)
chỉ bằng một dòng lệnh duy nhất:
    python src/run_all_pipeline.py
"""

# 1. Thư viện chuẩn của Python
import sys
import time
from pathlib import Path

# Đảm bảo thư mục `src/` luôn nằm trong sys.path dù gọi từ bất kỳ đâu
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 3. Mã của chính dự án
from bm_gnn.step0_tree_and_bm_simulator import run_step0_verification
from bm_gnn.step1_data_diagnostics_and_star import run_step1_pipeline
from bm_gnn.step2_phylolm_3pt_and_solvers import run_step2_pipeline
from bm_gnn.step3_gnn_architectures import verify_equivariance_and_invariance_properties
from bm_gnn.step3_train_and_ablation import run_step3_pipeline
from bm_gnn.step4_multiscale_and_synthesis import run_step4_pipeline


def main() -> None:
    """Chạy tuần tự toàn bộ pipeline từ Bước 0 đến Bước 4 và in báo cáo tổng hợp."""
    t_start = time.perf_counter()
    print("=" * 88)
    print("  KHỞI CHẠY TOÀN BỘ PIPELINE NGHIÊN CỨU BM-GNN TRÊN CÂY PHẢ HỆ HỌ KIẾN (148 LOÀI)")
    print("=" * 88)

    # Bước 0
    s0 = run_step0_verification()
    print(
        f"\n[Bước 0 Hoàn tất] Đồ thị |V|={s0['n_total_nodes']} nút, |E_dir|={s0['n_directed_edges']} cạnh | "
        f"ln det(S)={s0['logdet_S']:.6f} | vec11={s0['vec11']:.6f} | v_root={s0['v_root']:.6f}"
    )

    # Bước 1
    s1 = run_step1_pipeline()
    m0_mu = s1["star_baseline_full10000"]["mu"]
    m0_sg = s1["star_baseline_full10000"]["sigma_reml"]
    print(
        f"[Bước 1 Hoàn tất] Kiểm định {s1['n_samples_calib123']:,d} mẫu calib_123.csv (NaN={s1['nan_count']}) | "
        f"Star Baseline MSE(mu)={m0_mu['MSE']:.4f}, MSE(sigma)={m0_sg['MSE']:.4f}"
    )

    # Bước 2
    s2 = run_step2_pipeline()
    phy_mu = s2["phylolm_full10000_mu"]
    phy_sg = s2["phylolm_full10000_sigma"]
    print(
        f"[Bước 2 Hoàn tất] Đối chiếu phylolm vs. PIC/Cholesky sai số < {s2['max_diff_mu_phylolm_vs_cholesky']:.2e} | "
        f"t_REML={s2['t_reml_mean']:.6f} ± {s2['t_reml_se']:.6f} | "
        f"phylolm MSE(mu)={phy_mu['MSE']:.4f}, MSE(sigma)={phy_sg['MSE']:.5f}"
    )

    # Bước 3A
    s3a = verify_equivariance_and_invariance_properties()
    print(
        f"[Bước 3A Hoàn tất] Kiểm chứng Đẳng biến/Bất biến dịch chuyển: "
        f"sai số mu={s3a['translation_equivariance_max_error_mu']:.2e}, "
        f"sai số sigma={s3a['translation_invariance_max_error_sigma']:.2e}"
    )

    # Bước 3B
    s3b = run_step3_pipeline()
    df_te = s3b["test_1000_df"]
    print("\n[Bước 3B Hoàn tất] Bảng kết quả Ablation trên tập kiểm tra độc lập Test_1000 (1,000 cặp):")
    for _, r in df_te.iterrows():
        print(
            f"  - {r['Method']:26s}: MSE(mu)={r['mu_MSE']:.4f} (NRMSE_sig={r['mu_NRMSE_sigma']:.4f}, "
            f"PICP95={r['mu_PICP_95_pct']:.1f}%) | MSE(sigma)={r['sigma_MSE']:.5f} (R2={r['sigma_R2']:.4f})"
        )

    # Bước 4
    s4 = run_step4_pipeline()
    df_m = s4["multiscale_df"]
    sub_100k = df_m[df_m["N_pairs"] == 100000]
    print("\n[Bước 4 Hoàn tất] Kết quả tại quy mô lớn nhất N = 100,000 cặp tự mô phỏng:")
    for _, r in sub_100k.iterrows():
        print(
            f"  - {r['Method']:22s}: MSE(mu)={r['mu_MSE']:.4f} (NRMSE_sig={r['mu_NRMSE_sigma']:.4f}, "
            f"PICP95={r['mu_PICP_95_pct']:.2f}%) | MSE(sigma)={r['sigma_MSE']:.5f} (R2={r['sigma_R2']:.4f})"
        )

    elapsed = time.perf_counter() - t_start
    print("=" * 88)
    print(f"  HOÀN TẤT TOÀN BỘ PIPELINE TRONG {elapsed:.2f} GIÂY! MỌI KẾT QUẢ ĐÃ LƯU TẠI data/processed/")
    print("=" * 88)


if __name__ == "__main__":
    main()
