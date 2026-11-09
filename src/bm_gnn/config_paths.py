#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mô-đun cấu hình đường dẫn trung tâm (Centralized Path Configuration) cho `CodespaceGNN`.

Thiết kế Học máy Chuẩn (Machine Learning Experimental Protocol):
  - Tập Huấn luyện (`TRAIN_CALIB_CSV_PATH`): `data/raw/calib_123.csv` (10,000 mẫu).
  - Tập Kiểm tra Mô phỏng Cố định (`TEST_SIM_CSV_PATH`): `data/raw/test_sim_1000_fixed.csv`
    (1,000 mẫu độc lập ngoài mẫu, cố định hạt giống ngẫu nhiên `SEED_FIXED_TEST = 2026`).
"""

# 1. Thư viện chuẩn của Python
from pathlib import Path

# Hằng số đường dẫn toàn cục (UPPER_CASE theo chuẩn PEP 8)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
ARCH_DIAGRAMS_DIR: Path = (
    PROJECT_ROOT / "Ma_Nguon_va_Anh_Phuong_Phap_Kien_Truc_Hoc_Sau_GNN"
)
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Các tệp dữ liệu gốc (CHỈ ĐỌC trong data/raw/)
NWK_PATH: Path = RAW_DATA_DIR / "ants_tree_chuan.nwk"
META_PATH: Path = RAW_DATA_DIR / "ants_tree_chuan_metadata.json"
TRAIN_CALIB_CSV_PATH: Path = RAW_DATA_DIR / "calib_123.csv"
CALIB_CSV_PATH: Path = TRAIN_CALIB_CSV_PATH
TEST_SIM_CSV_PATH: Path = RAW_DATA_DIR / "test_sim_1000_fixed.csv"

# Biểu đồ Scatter Plot Chủ đạo (2-panel Predicted vs. True cho 1,000 mẫu Test x 3 phương pháp)
FLAGSHIP_SCATTER_PATH: Path = (
    FIGURES_DIR / "flagship_test1000_pred_vs_true_mu_sigma_scatter.png"
)

# Hằng số toán học & cấu hình thực nghiệm Học máy
SEED_FIXED_TEST: int = 2026
NUM_TRAIN_SAMPLES: int = 10000
NUM_TEST_SAMPLES: int = 1000
NUM_LEAVES: int = 148
NUM_INTERNAL_NODES: int = 147
NUM_TOTAL_NODES: int = 295
NUM_TREE_BRANCHES: int = 294
NUM_DIRECTED_EDGES: int = 588
ROOT_NODE_ID: int = 294
PRIOR_MU_MIN: float = -5.0
PRIOR_MU_MAX: float = 5.0
PRIOR_SIGMA_MIN: float = 0.1
PRIOR_SIGMA_MAX: float = 5.0


def ensure_output_directories() -> None:
    """Đảm bảo các thư mục đầu ra `data/processed/` và `figures/` luôn tồn tại."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_output_directories()
    print("=== KIỂM TRA CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN (CodespaceGNN) ===")
    print(f"  * PROJECT_ROOT         : {PROJECT_ROOT}")
    print(f"  * TRAIN_CALIB_CSV_PATH : {TRAIN_CALIB_CSV_PATH.name} (10,000 mẫu Train - Tồn tại: {TRAIN_CALIB_CSV_PATH.exists()})")
    print(f"  * TEST_SIM_CSV_PATH    : {TEST_SIM_CSV_PATH.name} (1,000 mẫu Test cố định - Tồn tại: {TEST_SIM_CSV_PATH.exists()})")
    print(f"  * FLAGSHIP_SCATTER     : {FLAGSHIP_SCATTER_PATH.name}")
