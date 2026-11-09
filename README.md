# CodespaceGNN — Ước Lượng Đồng Thời Hai Tham Số \$(\\mu, \\sigma)\$ Của Mô Hình Chuyển Động Brown Trên Cây Phả Hệ Họ Kiến (*Formicidae*, \$n \= 148\$ Loài)

## 1\. Dự án này làm gì?

Dự án này giải quyết bài toán ngược ước lượng đồng thời trạng thái gốc tổ tiên \$\\mu \\in \[-5, 5\]\$ và tốc độ khuếch tán tiến hóa \$\\sigma \\in \[0.1, 5.0\]\$ của quá trình Chuyển động Brown (*Brownian Motion — BM SDE*) khi chỉ quan sát một vector tính trạng liên tục \$\\mathbf{x} \\in \\mathbb{R}^{148}\$ tại \$n \= 148\$ loài kiến hiện sinh trên cây phả hệ định thời \$\|V\| \= 295\$ nút, \$\|E\| \= 294\$ nhánh.

Theo **Giao thức Thực nghiệm Học máy Chuẩn (Machine Learning Experimental Protocol)**:

- **Tập Huấn luyện (`Train Set`, \$N\_{\\mathrm{train}} \= 10{,}000\$ mẫu):** Sử dụng toàn bộ \$10{,}000\$ dòng của tệp `data/raw/calib_123.csv` để huấn luyện các toán tử truyền thông điệp đồ thị, trọng số cắt tỉa và bộ siêu tham số tiên nghiệm/cổng co rút \$(\\gamma\_\\mu, \\gamma\_\\sigma)\$.  
- **Tập Kiểm tra Độc lập Cố định (`Fixed Simulated Test Set`, \$N\_{\\mathrm{test}} \= 1{,}000\$ mẫu):** Sử dụng tập dữ liệu mô phỏng cố định `data/raw/test_sim_1000_fixed.csv` (`SEED_FIXED_TEST = 2026`, sinh từ cùng quá trình ngẫu nhiên \$\\mu\_k \\sim \\mathcal{U}(-5, 5)\$, \$\\ln\\sigma\_k \\sim \\mathcal{U}(\\ln 0.1, \\ln 5.0)\$, \$\\mathbf{x}*k \\sim \\mathcal{N}*{148}(\\mu\_k\\mathbf{1}, \\sigma\_k^2\\mathbf{S})\$) hoàn toàn tách biệt với tập huấn luyện.  
- **Biểu đồ Chủ đạo (`figures/flagship_test1000_pred_vs_true_mu_sigma_scatter.png`):** Gồm **2 biểu đồ Scatter Plot (Predicted Value vs. True Value có đường phân tách \$y \= x\$)** cho \$\\mu\$ và \$\\sigma\$, hiển thị đồng thời cả **\$1{,}000\$ giá trị ước lượng** cho cả **3 phương pháp chủ lực**: `phylolm` (Tiêu chuẩn Vàng cổ điển), `PhyloGNN_M5A` (GNN Đẳng biến Không chệch NLL, khớp `phylolm` với \$r \= 1.000000\$) và `PhyloGNN_M5B` (GNN Co rút Hậu nghiệm Bayes MMSE, giảm **\$-29.95%\$** \$\\mathrm{MSE}(\\hat{\\mu})\$ từ \$0.5453 \\to 0.3820\$ và giảm **\$-9.81%\$** \$\\mathrm{MSE}(\\hat{\\sigma})\$ từ \$0.01131 \\to 0.01020\$).

## 2\. Dữ liệu lấy từ đâu, ngày nào?

Toàn bộ dữ liệu gốc được lưu trữ ở chế độ **CHỈ ĐỌC (bất khả xâm phạm)** tại thư mục `data/raw/`:

- `data/raw/ants_tree_chuan.nwk`: Cây phả hệ phân tử định thời chuẩn của \$n \= 148\$ loài kiến hiện sinh (*Formicidae*, Moreau et al., *Science*, 2006), chiều cao siêu mels \$T\_i \= 1.0\$.  
- `data/raw/ants_tree_chuan_metadata.json`: Tệp ánh xạ chính xác thứ tự 148 loài lá (`0..147`).  
- `data/raw/calib_123.csv`: Tập dữ liệu huấn luyện chuẩn gồm \$N\_{\\mathrm{train}} \= 10{,}000\$ mẫu mô phỏng độc lập (\$10{,}000 \\times 150\$ cột).  
- `data/raw/test_sim_1000_fixed.csv`: Tập dữ liệu kiểm tra cố định gồm \$N\_{\\mathrm{test}} \= 1{,}000\$ mẫu mô phỏng độc lập ngoài mẫu (\$1{,}000 \\times 150\$ cột, cố định hạt giống `SEED_FIXED_TEST = 2026`).

## 3\. Cài đặt và chạy thế nào?

Mở thư mục `C:\Users\Admin\Desktop\CodespaceGNN` trong **Visual Studio Code** (`File -> Open Folder...`), mở cửa sổ **PowerShell Terminal** (\`Ctrl \+ \`\`) và chạy 4 lệnh sau:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/run_all_pipeline.py
```

Để chạy bộ kiểm thử tự động (5 bài test toán học & giao thức Học máy):

```
python -m pytest -v
```

*(Xem hướng dẫn chi tiết từng bước trên VS Code Windows tại tệp [`HUONG_DAN_CHAY_TREN_MAY_LOCAL_VSCODE_VI.md`](http://HUONG_DAN_CHAY_TREN_MAY_LOCAL_VSCODE_VI.md)).*

## 4\. Kết quả chính nằm ở đâu?

- **Biểu đồ Scatter Plot Chủ đạo (`1,000 mẫu Test x 3 phương pháp`):** `figures/flagship_test1000_pred_vs_true_mu_sigma_scatter.png` (cùng bộ 6 biểu đồ `flagship_test1000_pred_vs_true_mu_sigma_scatter.png` và `advisor_fig0..fig3.png` trong `figures/`).  
- **Bảng dự đoán 1,000 mẫu Test của 3 phương pháp (`phylolm`, `GNN M5A`, `GNN M5B`):** `data/processed/test_sim_1000_predictions_phylolm_m5a_m5b.csv`.  
- **Bảng tổng hợp chỉ số đánh giá:** `data/processed/step3_comprehensive_benchmark_comparison.csv` và `data/processed/master_notebook_exact_metrics.json`.  
- **Sơ đồ kiến trúc học sâu GNN & TikZ (`Hình 6`, `Hình K1` \$\\to\$ `K8`):** `Ma_Nguon_va_Anh_Phuong_Phap_Kien_Truc_Hoc_Sau_GNN/`.  
- **Báo cáo tương tác Jupyter Notebook:** `notebooks/BM_GNN_Ant_Phylogeny_Master_Pipeline_VI.ipynb`.
