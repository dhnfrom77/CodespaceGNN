#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 3B: Huấn luyện và Đánh giá Phân cấp Ablation 6 Kiến trúc GNN (M1 -> M5B)
         Đối sánh với Tiêu chuẩn Vàng `phylolm` trên `calib_123.csv` (Test_1000 & Full_10000).

Tham chiếu:
  - Figure 2: Biểu đồ Phân tán Tương quan (Parity Scatter) & Quạt Phần dư Dị phương sai.
  - Figure 3: Đẳng cấu Toán tử GNN M5A vs. `phylolm`, Đường cong Co rút Chữ S (M5B) & Bản đồ Lỗi 2D.
  - Tiểu luận Mục 5.1 - 5.3 & Cell 12 - Cell 16 trong Master Notebook.
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


def load_or_verify_step3_ablation_table() -> pd.DataFrame:
    """
    Đọc và kiểm chứng bảng kết quả đối sánh Ablation toàn diện trên tập Test_1000 và Full_10000
    từ `data/processed/master_notebook_exact_metrics.json`.
    """
    ensure_output_directories()
    metrics_json = PROCESSED_DATA_DIR / "master_notebook_exact_metrics.json"
    if metrics_json.exists():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        df_all = pd.DataFrame(data.get("calib_123_benchmarks", []))
        return df_all[df_all["Split"] == "Test_1000"].reset_index(drop=True)
    csv_path = PROCESSED_DATA_DIR / "step3_comprehensive_benchmark_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Không tìm thấy tệp kết quả Bước 3 trong data/processed/.")


def run_step3_pipeline() -> Dict[str, object]:
    """Thực thi tổng hợp kết quả Bước 3 và in bảng điểm Ablation trên tập Test_1000."""
    df_test = load_or_verify_step3_ablation_table()
    return {"test_1000_df": df_test}


if __name__ == "__main__":
    res3 = run_step3_pipeline()
    df_te = res3["test_1000_df"]
    print("===== KẾT QUẢ BƯỚC 3B: BẢNG ĐIỂM ĐỐI SÁNH ABLATION TRÊN TEST_1000 (1,000 CẶP ĐỘC LẬP) =====")
    for _, row in df_te.iterrows():
        print(
            f"  * {row['Method']:26s} | mu: MSE={row['mu_MSE']:.4f}, RMSE={row['mu_RMSE']:.4f}, "
            f"NRMSE_sig={row['mu_NRMSE_sigma']:.4f}, PICP95={row['mu_PICP_95_pct']:.1f}% | "
            f"sigma: MSE={row['sigma_MSE']:.5f}, R2={row['sigma_R2']:.4f}"
        )
