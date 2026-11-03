# CodespaceGNN — Ước Lượng Đồng Thời Hai Tham Số $(\mu, \sigma)$ Của Mô Hình Chuyển Động Brown Trên Cây Phả Hệ Họ Kiến (*Formicidae*, $n = 148$ Loài)

## 1. Dự án này làm gì?
Dự án này giải quyết bài toán ngược ước lượng đồng thời trạng thái gốc tổ tiên $\mu \in [-5, 5]$ và tốc độ khuếch tán tiến hóa $\sigma \in [0.1, 5.0]$ của quá trình Chuyển động Brown (*Brownian Motion — BM SDE*) khi chỉ quan sát một vector tính trạng liên tục $\mathbf{x} \in \mathbb{R}^{148}$ tại $n = 148$ loài kiến hiện sinh trên cây phả hệ định thời $|V| = 295$ nút, $|E| = 294$ nhánh. Dự án cài đặt từ gốc thuật toán cắt tỉa 3 điểm tuyến tính $\mathcal{O}(n)$ của gói `phylolm` (Ho & Ané, 2014) làm Tiêu chuẩn Vàng cổ điển và thiết kế phân cấp 6 kiến trúc Mạng Nơ-ron Đồ thị ($\mathcal{M}_1 \to \mathcal{M}_{5B}$), chứng minh `PhyloGNN_M5A_UnbiasedNLL` hội tụ sát tuyệt đối `phylolm` ($r = 1.000000$) còn `PhyloGNN_M5B_BayesMMSE` giảm $30.18\%$ $\mathrm{MSE}(\hat{\mu})$ và $13.11\%$ $\mathrm{MSE}(\hat{\sigma})$ trên tập kiểm tra độc lập.

## 2. Dữ liệu lấy từ đâu, ngày nào?
Toàn bộ dữ liệu gốc được lưu trữ ở chế độ **CHỈ ĐỌC (bất khả xâm phạm)** tại thư mục `data/raw/`:
- `data/raw/ants_tree_chuan.nwk`: Cây phả hệ phân tử định thời chuẩn của $n = 148$ loài kiến hiện sinh (*Formicidae*) trích xuất từ công trình của Moreau và cộng sự (*Science*, 2006), được chuẩn hóa chiều cao siêu mels $T_i = 1.0$ (cung cấp ngày 15/10/2026).
- `data/raw/ants_tree_chuan_metadata.json`: Tệp ánh xạ chính xác thứ tự 148 loài lá giữa cây Newick và bảng dữ liệu (cung cấp ngày 15/10/2026).
- `data/raw/calib_123.csv`: Tập dữ liệu chuẩn gồm $M = 10{,}000$ mẫu mô phỏng độc lập $(\mu_k, \sigma_k, \mathbf{x}_k \in \mathbb{R}^{148})$ phục vụ huấn luyện, kiểm định chéo và đối sánh mô hình (cung cấp ngày 15/10/2026).

## 3. Cài đặt và chạy thế nào?
Mở thư mục `C:\Users\Admin\Desktop\CodespaceGNN` trong **Visual Studio Code** (`File -> Open Folder...`), mở cửa sổ **PowerShell Terminal** (`Ctrl + \``) và chạy 4 lệnh sau:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/run_all_pipeline.py
```

Để chạy kiểm thử tự động và kiểm tra chất lượng mã theo chuẩn PEP 8:
```powershell
python -m pytest
ruff format --check .
ruff check .
```

*(Xem hướng dẫn cầm tay chỉ việc từng bước trên VS Code Windows tại tệp [`HUONG_DAN_CHAY_TREN_MAY_LOCAL_VSCODE_VI.md`](HUONG_DAN_CHAY_TREN_MAY_LOCAL_VSCODE_VI.md)).*

## 4. Kết quả chính nằm ở đâu?
- **Các bảng số liệu kết quả thực nghiệm (`.csv`, `.json`):** Nằm tại `data/processed/` (bao gồm ma trận $\mathbf{S}$, trọng số $\mathbf{w}_{\mathrm{GLS}}$, bảng đối sánh Ablation `step3_comprehensive_benchmark_comparison.csv`, bảng đa quy mô `step4_prime_multiscale_summary_metrics.csv`, và tệp tổng hợp `master_notebook_exact_metrics.json`).
- **Các biểu đồ chẩn đoán thực nghiệm (`Figure 0A`, `Figure 0B-C`, `Figure 1` $\to$ `Figure 5`):** Nằm tại `figures/`.
- **Các sơ đồ kiến trúc học sâu GNN & phương pháp toán học (`Hình 6`, `Hình K1` $\to$ `Hình K8` kèm mã nguồn TikZ):** Nằm tại thư mục con `Ma_Nguon_va_Anh_Phuong_Phap_Kien_Truc_Hoc_Sau_GNN/`.
- **Báo cáo tương tác Jupyter Notebook:** Nằm tại `notebooks/BM_GNN_Ant_Phylogeny_Master_Pipeline_VI.ipynb`.
