#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mô-đun cấu hình đường dẫn trung tâm (Centralized Path Configuration).

Tuân thủ quy chuẩn PEP 8 và cấu trúc dự án Khoa học Dữ liệu:
- Tự động xác định thư mục gốc `CodebaseGNN` qua `pathlib.Path(__file__).resolve()`.
- Đảm bảo nguyên tắc: `data/raw/` chỉ đọc (bất khả xâm phạm), mọi kết quả ghi ra
  `data/processed/` và hình vẽ xuất ra `figures/`.
"""

# 1. Thư viện chuẩn của Python
from pathlib import Path

# Hằng số đường dẫn toàn cục (UPPER_CASE theo chuẩn PEP 8)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
ARCH_DIAGRAMS_DIR: Path = PROJECT_ROOT / "Ma_Nguon_va_Anh_Phuong_Phap_Kien_Truc_Hoc_Sau_GNN"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Ba tệp dữ liệu gốc (CHỈ ĐỌC trong data/raw/)
NWK_PATH: Path = RAW_DATA_DIR / "ants_tree_chuan.nwk"
META_PATH: Path = RAW_DATA_DIR / "ants_tree_chuan_metadata.json"
CALIB_CSV_PATH: Path = RAW_DATA_DIR / "calib_123.csv"

# Các hằng số toán học của cây phả hệ Họ Kiến (Formicidae, n = 148 loài)
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
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_output_directories()
    print("=== KIỂM TRA CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN (CodebaseGNN) ===")
    print(f"  * PROJECT_ROOT       : {PROJECT_ROOT}")
    print(f"  * RAW_DATA_DIR       : {RAW_DATA_DIR} (Tồn tại: {RAW_DATA_DIR.exists()})")
    print(f"  * PROCESSED_DATA_DIR : {PROCESSED_DATA_DIR} (Tồn tại: {PROCESSED_DATA_DIR.exists()})")
    print(f"  * FIGURES_DIR        : {FIGURES_DIR} (Tồn tại: {FIGURES_DIR.exists()})")
    print(f"  * ARCH_DIAGRAMS_DIR  : {ARCH_DIAGRAMS_DIR} (Tồn tại: {ARCH_DIAGRAMS_DIR.exists()})")
    print(f"  * Tệp cây Newick     : {NWK_PATH.name} (Tồn tại: {NWK_PATH.exists()})")
    print(f"  * Tệp Metadata JSON  : {META_PATH.name} (Tồn tại: {META_PATH.exists()})")
    print(f"  * Tệp calib_123.csv  : {CALIB_CSV_PATH.name} (Tồn tại: {CALIB_CSV_PATH.exists()})")
