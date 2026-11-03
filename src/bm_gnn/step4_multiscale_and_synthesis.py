#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 4: Thực nghiệm Mở rộng Đa Quy mô Ngoài mẫu (N in {100, 1,000, 10,000, 100,000} Cặp Tự Mô phỏng),
        Kiểm chứng Định luật Hội tụ Trọng số O(N^{-1/2}) và Hiệu chuẩn Khoảng Dự đoán Cramér-Rao 95%.

Tham chiếu:
  - Figure 4: Lưới 8 Biểu đồ Phân tán Đa Quy mô (N = 100 -> 100,000).
  - Figure 5: Đường cong Mở rộng Quy mô Mẫu, Hội tụ Trọng số O(N^{-1/2}) & Độ phủ PICP 95%.
  - Tiểu luận Mục 6.1 - 6.3 & Cell 17 - Cell 23 trong Master Notebook.
"""

# 1. Thư viện chuẩn của Python
import json
from typing import Dict

# 2. Thư viện bên thứ ba
import pandas as pd

# 3. Mã của chính dự án
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_gnn.config_paths import PROCESSED_DATA_DIR, ensure_output_directories


def load_multiscale_benchmark_table() -> pd.DataFrame:
    """Đọc bảng kết quả thực nghiệm đa quy mô N = 100 -> 100,000 từ `data/processed/`."""
    ensure_output_directories()
    metrics_json = PROCESSED_DATA_DIR / "master_notebook_exact_metrics.json"
    if metrics_json.exists():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        return pd.DataFrame(data.get("multiscale_multipair_benchmarks", []))
    csv_path = PROCESSED_DATA_DIR / "step4_prime_multiscale_summary_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Không tìm thấy tệp kết quả đa quy mô trong data/processed/.")


def run_step4_pipeline() -> Dict[str, pd.DataFrame]:
    """Thực thi tổng hợp kết quả đa quy mô Bước 4."""
    df_multi = load_multiscale_benchmark_table()
    return {"multiscale_df": df_multi}


if __name__ == "__main__":
    res4 = run_step4_pipeline()
    df_m = res4["multiscale_df"]
    print("===== KẾT QUẢ BƯỚC 4: THỰC NGHIỆM ĐA QUY MÔ N = 100 -> 100,000 CẶP TỰ MÔ PHỎNG =====")
    for n_scale in [100, 1000, 10000, 100000]:
        print(f"\n--- QUY MÔ N = {n_scale:,d} CẶP THAM SỐ (mu_k, sigma_k) ---")
        sub = df_m[df_m["N_pairs"] == n_scale]
        for _, row in sub.iterrows():
            print(
                f"  * {row['Method']:22s} | mu: MSE={row['mu_MSE']:.4f}, NRMSE_sig={row['mu_NRMSE_sigma']:.4f}, "
                f"PICP95={row['mu_PICP_95_pct']:.1f}% | sigma: MSE={row['sigma_MSE']:.5f}, R2={row['sigma_R2']:.4f}"
            )
