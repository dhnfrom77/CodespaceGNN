# CẨM NANG HƯỚNG DẪN TỔ CHỨC VÀ CHẠY TỪNG BƯỚC DỰ ÁN `CodespaceGNN` TRÊN MÁY LOCAL (VISUAL STUDIO CODE — WINDOWS)

* **Thư mục dự án trên máy của bạn:** `C:\Users\Admin\Desktop\CodespaceGNN`  
    
* **Công cụ sử dụng:** Visual Studio Code (VS Code), Python 3.11+, Windows PowerShell, Git & GitHub  
    
* **Tài liệu chuẩn tham chiếu:** Được biên soạn bám sát 100% quy chuẩn kỹ nghệ phần mềm và quy trình làm việc của 3 tài liệu:  
    
  1. `Lab01_Moi_truong_va_cu_phap.pdf` *(Bài thực hành 1 — Môi trường làm việc, `.venv`, Kernel VS Code và các hộp Mốc kiểm)*  
       
  2. `Bai01_He_sinh_thai_Python.pdf` *(Bài 1 — Cấu trúc cây thư mục chuẩn `data/raw`, `data/processed`, `src/`, `notebooks/`, `tests/`, `figures/`, PEP 8 và `ruff`)*  
       
  3. `Bai01_PhuLuc_Git_GitHub.pdf` *(Phụ lục Bài 1 — Quản lý phiên bản với `git init`, `.gitignore`, `git add`, `git commit`, `git push` lên GitHub)*

## PHẦN 1\. TỔNG QUAN CẤU TRÚC THƯ MỤC CHUẨN CỦA `CodespaceGNN`

Thay vì dồn toàn bộ hàng trăm dòng định nghĩa lớp, hàm giải tích và hàm vẽ đồ thị vào trong một tệp `.ipynb` duy nhất, toàn bộ dự án trong thư mục `C:\Users\Admin\Desktop\CodespaceGNN` đã được tái cấu trúc theo đúng **3 nguyên tắc vàng của dự án Khoa học Dữ liệu**:

1. **Nguyên tắc 1 (`data/raw/` là bất khả xâm phạm):** Bốn tệp dữ liệu chuẩn trong `data/raw/` gồm cây phả hệ (`ants_tree_chuan.nwk`, `ants_tree_chuan_metadata.json`), **Tập Huấn luyện \$10,000\$ mẫu (`calib_123.csv`)** và **Tập Kiểm tra Mô phỏng Độc lập Cố định \$1,000\$ mẫu (`test_sim_1000_fixed.csv`, `SEED_FIXED_TEST = 2026`)** nằm ở chế độ **CHỈ ĐỌC**. Không một dòng mã nào được phép ghi đè lên `data/raw/`. Mọi bảng kết quả sinh ra đều được ghi vào `data/processed/`.  
     
2. **Nguyên tắc 2 (`data/processed/` và `figures/` hoàn toàn tái lập được):** Toàn bộ 8 tệp kết quả `.csv`/`.json` chuẩn trong `data/processed/` và các hình vẽ trong `figures/` đều được sinh ra từ mã nguồn Python.  
     
3. **Nguyên tắc 3 (Tách mã lõi tái sử dụng trong `src/bm_gnn/` khỏi `notebooks/`):** Toàn bộ các lớp đồ thị, thuật toán cắt tỉa 3 điểm `phylolm`, 6 kiến trúc mạng GNN (\$\\mathcal{M}*1 \\to \\mathcal{M}*{5B}\$) và quy trình thực nghiệm theo từng bước được tách thành các mô-đun `.py` độc lập trong `src/bm_gnn/`. Mỗi tệp `.py` đều có khối `if __name__ == "__main__":` ở cuối tệp để **vừa chạy trực tiếp được từ dòng lệnh, vừa `import` được vào Notebook và bộ kiểm thử `tests/`**. Thư mục chứa toàn bộ ảnh kiến trúc và mã nguồn TikZ (`Ma_Nguon_va_Anh_Phuong_Phap_Kien_Truc_Hoc_Sau_GNN/`) cũng được đặt gọn làm một thư mục con ngay bên trong `CodespaceGNN/`.

```
C:\Users\Admin\Desktop\CodespaceGNN\
│
├── data\
│   ├── raw\                                     # [CHỈ ĐỌC - BẤT KHẢ XÂM PHẠM] Dữ liệu gốc ban đầu
│   │   ├── ants_tree_chuan.nwk                  # Cây phả hệ định thời 148 loài kiến (295 nút, 294 nhánh)
│   │   ├── ants_tree_chuan_metadata.json        # Metadata ánh xạ tên và chỉ số 148 loài lá (0..147)
│   │   ├── calib_123.csv                        # [TẬP TRAIN] Toàn bộ 10,000 mẫu dùng để huấn luyện GNN & cổng co rút
│   │   └── test_sim_1000_fixed.csv              # [TẬP TEST CỐ ĐỊNH] 1,000 mẫu mô phỏng độc lập (SEED=2026) để đánh giá
│   │
│   └── processed\                               # [SINH TỪ MÃ] Đúng 8 bảng số liệu .csv và .json theo giao thức chuẩn
│       ├── ants_phylo_structure_matrix_S.csv
│       ├── ants_phylo_inverse_and_gls_weights.csv
│       ├── step3_ant_phylogeny_graph_nodes.csv
│       ├── step3_ant_phylogeny_graph_edges.csv
│       ├── test_sim_1000_predictions_phylolm_m5a_m5b.csv  # Bảng 1,000 dự đoán (mu, sigma) trên tập Test_Sim_1000
│       ├── step3_comprehensive_benchmark_comparison.csv   # Bảng điểm 8 phương pháp trên tập Test_Sim_1000 (8 dòng)
│       ├── step4_test1000_regime_and_boundary_analysis.csv
│       └── master_notebook_exact_metrics.json
│
├── src\                                         # [MÃ LÕI TÁI SỬ DỤNG] Tuân thủ chuẩn PEP 8 & thứ tự import 3 nhóm
│   ├── __init__.py
│   ├── run_all_pipeline.py                      # Kịch bản điều phối chạy một mạch từ Bước 0 -> Bước 4
│   └── bm_gnn\                                  # Gói Python chính của dự án
│       ├── __init__.py                          # Xuất (export) các lớp và hàm cốt lõi
│       ├── config_paths.py                      # Định vị tự động đường dẫn dự án (pathlib.Path)
│       ├── step0_tree_and_bm_simulator.py       # Bước 0: Đọc Newick, dựng đồ thị 295 nút, ma trận S, Cholesky L_S & mô phỏng BM
│       ├── step1_data_diagnostics_and_star.py   # Bước 1: Kiểm toán calib_123.csv, hệ số kappa_T = 0.6151, Star M0 & Figure 0B-C
│       ├── step2_phylolm_3pt_and_solvers.py     # Bước 2: Cắt tỉa 3 điểm tuyến tính phylolm O(n), PIC, Cholesky GLS & kiểm định t ≈ 1.0
│       ├── step3_gnn_architectures.py           # Bước 3A: Tầng DirectedBottomUpTreeMPNNLayer & 6 kiến trúc GNN (M1 -> M5B)
│       ├── step3_train_and_ablation.py          # Bước 3B: Huấn luyện trên Train_10000 & đánh giá độc lập trên Test_Sim_1000
│       └── step4_multiscale_and_synthesis.py    # Bước 4: Phân tích phân tầng 4 chế độ sigma_k & vùng biên mu_k trên Test_Sim_1000
│
├── notebooks\                                   # [KHÁM PHÁ & TRÌNH BÀY] Chỉ import từ src/bm_gnn/
│   └── BM_GNN_Ant_Phylogeny_Master_Pipeline_VI.ipynb
│
├── tests\                                       # [KIỂM THỬ TỰ ĐỘNG] Kiểm chứng toán học bằng pytest / unittest
│   └── test_bm_gnn_pipeline.py
│
├── figures\                                     # [HÌNH VẼ THỰC NGHIỆM] Biểu đồ Scatter chủ đạo & các biểu đồ chẩn đoán
│   ├── flagship_test1000_pred_vs_true_mu_sigma_scatter.png # [BIỂU ĐỒ CHỦ ĐẠO] 2 Scatter Plot (mu, sigma) x 1,000 mẫu x 3 phương pháp
│   ├── advisor_fig0_ultra_tall_full_148_species_phylogeny.png
│   ├── advisor_fig0_bc_horizontal_weights_and_traits.png
│   ├── advisor_fig1_data_and_tree_diagnostics.png
│   ├── advisor_fig2_simultaneous_parity_and_residual_scatter.png
│   └── advisor_fig3_concordance_and_2d_error_scatter.png
│
├── Ma_Nguon_va_Anh_Phuong_Phap_Kien_Truc_Hoc_Sau_GNN\  # [THƯ MỤC CON ẢNH KIẾN TRÚC & TIKZ]
│   ├── advisor_fig_gnn_architecture_schematic.{png,pdf,tex}
│   ├── fig_method_01_spatiotemporal_bm_sde_and_mrca_covariance_3d.{png,pdf,tex}
│   ├── fig_method_02_hilbert_space_orthogonal_decomposition_147_plus_1_dof.{png,pdf,tex}
│   ├── fig_method_03_phylolm_linear_time_three_point_pruning_flowchart.{png,pdf,tex}
│   ├── fig_gnn_04_phylogenetic_tree_to_pyg_bidirected_graph_tensors.{png,pdf,tex}
│   ├── fig_gnn_05_six_tier_architectural_ablation_matrix_m1_to_m5b.{png,pdf,tex}
│   ├── fig_gnn_06_directed_bottom_up_tree_mpnn_cell_microarchitecture.{png,pdf,tex}
│   ├── fig_gnn_07_m5a_dual_stream_equivariant_phylognn_and_nll_loss.{png,pdf,tex}
│   └── fig_gnn_08_m5b_amortized_empirical_bayes_shrinkage_and_cohort_updating.{png,pdf,tex}
│
├── requirements.txt                             # Danh sách thư viện ghim phiên bản chính xác
├── .gitignore                                   # Chặn .venv/, __pycache__/ không đưa vào Git
├── README.md                                    # Tóm tắt dự án trả lời 4 câu hỏi chuẩn
└── HUONG_DAN_CHAY_TREN_MAY_LOCAL_VSCODE_VI.md   # Tài liệu hướng dẫn từng bước này

```

## PHẦN 2\. MỞ DỰ ÁN VÀ DỰNG MÔI TRƯỜNG ẢO `.venv` TRONG VS CODE

### Bước 2.1. Mở đúng thư mục dự án trong VS Code (Luôn mở thư mục, không mở tệp lẻ)

1. Khởi động **Visual Studio Code**.  
     
2. Kiểm tra hai tiện ích mở rộng bắt buộc của Microsoft: bấm biểu tượng **Extensions** ở thanh bên trái (hoặc nhấn `Ctrl + Shift + X`), tìm và đảm bảo đã cài đặt **Python** và **Jupyter**.  
     
3. Chọn menu **File** \$\\to\$ **Open Folder...**.  
     
4. Đi tới đường dẫn `C:\Users\Admin\Desktop\CodespaceGNN`, bấm một lần vào thư mục `CodespaceGNN` rồi bấm **Select Folder**.  
     
5. Nếu VS Code hiện hộp thoại hỏi *"Do you trust the authors of the files in this folder?"*, bấm **Yes, I trust the authors**.

>   
> **\[MỐC KIỂM 1 — Khung Explorer bên trái VS Code\]** Khung *Explorer* bên trái bây giờ phải hiện dòng chữ in hoa **`CODESPACEGNN`**, và ngay bên dưới nó là các thư mục con `data`, `figures`, `Ma_Nguon_va_Anh_Phuong_Phap_Kien_Truc_Hoc_Sau_GNN`, `notebooks`, `src`, `tests` cùng các tệp `requirements.txt`, `README.md`. *Lưu ý:* Nếu bạn thấy `CodespaceGNN` nằm lồng bên trong một thư mục khác, hãy chọn lại `File -> Open Folder...` trỏ đúng vào `C:\Users\Admin\Desktop\CodespaceGNN` để terminal tự động đứng đúng ở gốc dự án và VS Code nhận diện được `.venv`.

### Bước 2.2. Kiểm tra bản Python toàn cục đang chạy

Mở cửa sổ Terminal ngay bên trong VS Code bằng cách chọn **Terminal** \$\\to\$ **New Terminal** (hoặc nhấn tổ hợp phím `` Ctrl + ` ``). Cửa sổ **PowerShell** sẽ hiện ở nửa dưới màn hình và dấu nhắc đang đứng sẵn tại `PS C:\Users\Admin\Desktop\CodespaceGNN>`. Gõ hai lệnh sau:

```
python --version
python -c "import sys; print(sys.executable)"

```

* Nếu máy báo không nhận lệnh `python`, hãy thử gõ `py -3 --version`.  
    
* Lệnh thứ hai sẽ in ra đường dẫn tới Python toàn cục của máy (ví dụ: `C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe`). Hãy ghi nhớ đường dẫn này — ngay sau khi kích hoạt môi trường ảo ở bước tiếp theo, đường dẫn này sẽ đổi sang `.venv`.

### Bước 2.3. Tạo và kích hoạt môi trường ảo `.venv` riêng cho dự án

Vẫn trong cửa sổ Terminal của VS Code tại `C:\Users\Admin\Desktop\CodespaceGNN`, gõ lệnh tạo môi trường ảo:

```
python -m venv .venv

```

* Nếu VS Code bật thông báo ở góc dưới bên phải: *"We noticed a new environment has been created. Do you want to select it for the workspace folder?"*, hãy bấm **Yes**.  
    
* Tạo xong **chưa dùng được ngay** — bạn bắt buộc phải **kích hoạt (activate)** môi trường ảo trong Terminal:

```
.venv\Scripts\Activate.ps1

```

> **\[Xử lý lỗi thường gặp trên Windows PowerShell: *"running scripts is disabled on this system"*\]** Mặc định Windows PowerShell có thể chặn chạy tệp kịch bản `Activate.ps1`. Nếu gặp dòng báo lỗi màu đỏ này, bạn chỉ cần chạy đúng một lần lệnh cấp quyền dưới đây rồi chạy lại lệnh kích hoạt:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1

```

> *(Nếu bạn đang dùng Command Prompt `cmd` thay vì PowerShell thì gõ: `.venv\Scripts\activate.bat`).*

Sau khi kích hoạt, đầu dòng lệnh sẽ có thêm tiền tố `(.venv)`. Tuy nhiên, **đừng chỉ tin vào dấu nhắc lệnh — hãy hỏi thẳng Python** bằng lệnh kiểm chứng:

```
python -c "import sys; print(sys.executable)"

```

> **\[MỐC KIỂM 2 (QUAN TRỌNG NHẤT) — Kiểm chứng `sys.executable`\]** Kết quả in ra trên màn hình bây giờ **bắt buộc phải chứa `.venv`** nằm bên trong thư mục `CodespaceGNN`:

```
C:\Users\Admin\Desktop\CodespaceGNN\.venv\Scripts\python.exe

```

> Nếu đường dẫn in ra **không chứa `.venv`**, tuyệt đối **đừng đi tiếp**, vì mọi thư viện cài ở bước sau sẽ bị cài nhầm vào Python toàn cục của máy chứ không vào dự án\!

### Bước 2.4. Cài đặt thư viện từ `requirements.txt` bằng `python -m pip`

Luôn luôn gõ **`python -m pip`** thay cho `pip` trần để chắc chắn 100% bạn đang gọi đúng trình cài đặt `pip` của bản Python bên trong `.venv`:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

*(Tùy chọn mở rộng)*: Dự án đã được thiết kế để toàn bộ pipeline giải tích, kiểm định thống kê, huấn luyện/đánh giá đẳng cấu toán tử và vẽ đồ thị chạy mượt mà ngay với các gói chuẩn trong `requirements.txt` (`numpy`, `pandas`, `scipy`, `matplotlib`). Nếu máy bạn muốn chạy thêm các lớp mạng `torch.nn.Module` và `MessagePassing` bằng GPU/CPU của PyTorch & PyTorch Geometric, bạn có thể cài thêm:

```
python -m pip install torch torch-geometric

```

> **\[MỐC KIỂM 3 — Kiểm tra các gói đã vào đúng `.venv`\]** Gõ lệnh kiểm tra đường dẫn nạp thư viện `numpy` và `pandas`:

```
python -c "import numpy, pandas; print(numpy.__file__); print(pandas.__file__)"

```

> Cả hai đường dẫn in ra đều phải bắt đầu bằng `C:\Users\Admin\Desktop\CodespaceGNN\.venv\Lib\site-packages\...`.

## PHẦN 3\. BỐN CÁCH CHẠY VÀ KHÁM PHÁ PIPELINE TRONG `CodespaceGNN`

Tất cả các thao tác dưới đây đều được thực hiện trong cửa sổ Terminal đã kích hoạt `(.venv)` đứng tại `C:\Users\Admin\Desktop\CodespaceGNN`.

### Cách 1 — Chạy từng tệp kịch bản `.py` theo từng bước (Hoặc chạy toàn bộ bằng 1 lệnh)

Mỗi tệp `.py` trong `src/bm_gnn/` đại diện cho một bước độc lập trong luồng nghiên cứu và đều có khối `if __name__ == "__main__":`. Để Python tìm thấy gói `bm_gnn` trong `src/`, bạn có thể chạy kịch bản điều phối tổng thể hoặc chạy từng bước như sau:

1. **Chạy một mạch toàn bộ Pipeline (từ Bước 0 đến Bước 4):**

```
python src/run_all_pipeline.py

```

2. **Hoặc chạy riêng lẻ từng bước một để quan sát và mổ xẻ kết quả:**

```
# Kiểm tra cấu hình đường dẫn dự án
python src/bm_gnn/config_paths.py

# Bước 0: Dựng đồ thị 295 nút, 588 cạnh có hướng, ma trận S (148x148), nhân tử Cholesky L_S, vec11 & v_root
python src/bm_gnn/step0_tree_and_bm_simulator.py

# Bước 1: Kiểm toán 10,000 mẫu calib_123.csv, hệ số suy giảm phương sai kappa_T = 0.6151 & Star Baseline (M0)
python src/bm_gnn/step1_data_diagnostics_and_star.py

# Bước 2: Cắt tỉa 3 điểm tuyến tính phylolm O(n), đối chiếu PIC / Cholesky GLS (< 10^-15) & kiểm định t ≈ 1.0
python src/bm_gnn/step2_phylolm_3pt_and_solvers.py

# Bước 3A: Kiểm chứng toán học tính Đẳng biến dịch chuyển của mu_hat và Bất biến dịch chuyển của sigma_hat
python src/bm_gnn/step3_gnn_architectures.py

# Bước 3B: Huấn luyện 6 kiến trúc GNN (M1 -> M5B) trên Train_10000 & đánh giá độc lập trên Test_Sim_1000
python src/bm_gnn/step3_train_and_ablation.py

# Bước 4: Phân tích phân tầng theo 4 chế độ tốc độ tiến hóa sigma_k và vùng biên mu_k trên Test_Sim_1000
python src/bm_gnn/step4_multiscale_and_synthesis.py

```

   *(Mẹo nhỏ trên VS Code)*: Nếu bạn muốn bấm nút **Run (**\$\\triangleright\$**)** ở góc trên bên phải cửa sổ VS Code, hãy nhìn xuống thanh trạng thái góc dưới bên phải cửa sổ: nếu chưa hiện `.venv`, hãy bấm vào đó (hoặc nhấn `Ctrl + Shift + P` \$\\to\$ gõ **`Python: Select Interpreter`**) và chọn **`Python 3.11.x ('.venv': venv)`**.

### Cách 2 — Chạy bộ kiểm thử tự động (`pytest` / `unittest`)

Để chứng minh bằng lệnh rằng toàn bộ cấu trúc đồ thị 295 nút, thuật toán cắt tỉa 3 điểm `phylolm`, tính chất đẳng biến/bất biến dịch chuyển của mạng `PhyloGNN_M5A` và công thức co rút Chuẩn Cụt của `PhyloGNN_M5B` đều hoạt động chính xác tuyệt đối:

```
python -m pytest -v

```

*(Hoặc dùng mô-đun có sẵn của Python: `python -m unittest discover -s tests -v`).*

> **\[MỐC KIỂM 4 — Kết quả kiểm thử tự động\]** Màn hình phải in ra **`5 passed`** (hoặc **`OK`** cho cả 5 bài kiểm thử toán học `test_01` \$\\to\$ `test_05`, bao gồm kiểm chứng giao thức `Train_10000` vs. `Test_Sim_1000`).

### Cách 3 — Khám phá tương tác nhanh bằng `IPython` (`type`, `dir`, `help`, `%timeit`)

Khi bạn muốn tra cứu nhanh chữ ký hàm, đọc docstring hoặc đo thời gian chạy thực tế của thuật toán cắt tỉa 3 điểm `phylolm` mà không cần sửa tệp:

```
ipython

```

Khi dấu nhắc đổi thành `In [1]:`, hãy gõ thử các lệnh sau:

```
from bm_gnn import FromScratchPhylogenyBuilder, run_phylolm_3pt_pruning_from_scratch
import numpy as np

# Xem tài liệu hướng dẫn (docstring) của lớp và hàm bằng dấu ? hoặc help()
FromScratchPhylogenyBuilder?
run_phylolm_3pt_pruning_from_scratch?

# Khởi tạo đồ thị cây 148 loài kiến và kiểm tra kiểu / thuộc tính (bộ ba type - dir - help)
builder = FromScratchPhylogenyBuilder()
type(builder)
print("vec11 =", builder.vec11, "| v_root =", builder.v_root)

# Đo thời gian cắt tỉa 3 điểm tuyến tính phylolm trên 1,000 cây (Quy tắc 2: Đo trước khi tối ưu!)
X_test = builder.simulate_forward_bm_preorder(mu=0.0, sigma=1.5, n_replicates=1000)
%timeit run_phylolm_3pt_pruning_from_scratch(builder, X_test)

exit

```

### Cách 4 — Chạy tệp báo cáo `BM_GNN_Ant_Phylogeny_Master_Pipeline_VI.ipynb` trong VS Code

1. Ở khung *Explorer* bên trái VS Code, mở thư mục `notebooks` và bấm vào tệp **`BM_GNN_Ant_Phylogeny_Master_Pipeline_VI.ipynb`**.  
     
2. Nhìn lên **góc trên bên phải** của cửa sổ Notebook, bấm vào nút **Select Kernel**.  
     
3. Chọn **Python Environments...**, sau đó chọn dòng có chứa **`.venv`** (ví dụ: `.venv (Python 3.11.x) — C:\Users\Admin\Desktop\CodespaceGNN\.venv\Scripts\python.exe`).  
     
   * *Mẹo dự phòng:* Nếu danh sách chưa hiện `.venv`, hãy mở Terminal chạy lệnh đăng ký kernel riêng rồi chọn trong mục *Jupyter Kernel...*:

```
python -m ipykernel install --user --name codespace-gnn --display-name "Python (.venv CodespaceGNN)"

```

4. Bấm vào ô mã đầu tiên (**Ô mã `[1]`**) và nhấn `Shift + Enter` để chạy.

>   
> **\[MỐC KIỂM 5 — Kiểm tra trạng thái sạch bằng `Restart` rồi `Run All`\]**  
> 

> 1. Dòng đầu tiên dưới Ô mã `[1]` phải in ra: `[Mốc kiểm Kernel Python] sys.executable = C:\Users\Admin\Desktop\CodespaceGNN\.venv\Scripts\python.exe`  
   >   
> 2. Để đảm bảo không có trạng thái ẩn (*hidden state*), hãy bấm nút **Restart** trên thanh công cụ phía trên Notebook, sau đó bấm **Run All**.  
   >   
> 3. Toàn bộ 9 ô mã trong Notebook phải chạy thông suốt từ trên xuống dưới, số thứ tự bên trái các ô mã tăng liên tục từ **`[1]`, `[2]`, `[3]` ... đến `[9]`**, hiển thị đầy đủ các sơ đồ kiến trúc (`Hình 6`, `Hình K1` \$\\to\$ `K8`) và các biểu đồ thực nghiệm (`Figure 0A`, `Figure 0B-C`, `Figure 1` \$\\to\$ `Figure 5`).

## PHẦN 4\. KIỂM TRA CHẤT LƯỢNG MÃ THEO CHUẨN PEP 8 VỚI `ruff`

Toàn bộ các mô-đun `.py` trong `src/` và `tests/` được viết theo chuẩn **PEP 8**:

* Tên hàm và biến dùng `snake_case` (`run_phylolm_3pt_pruning_from_scratch`, `kappa_tree`).  
    
* Tên lớp dùng `CapWords` (`PhyloTreeNode`, `FromScratchPhylogenyBuilder`).  
    
* Hằng số toàn cục dùng `UPPER_CASE` (`PROJECT_ROOT`, `RAW_DATA_DIR`, `NUM_LEAVES`).  
    
* Các lệnh `import` ở đầu mỗi tệp được chia đúng 3 nhóm cách nhau 1 dòng trống: *(1) Thư viện chuẩn của Python* \$\\to\$ *(2) Thư viện bên thứ ba* \$\\to\$ *(3) Mã của chính dự án (`bm_gnn`)*.

Trước mỗi lần bàn giao hoặc lưu phiên bản mới sau khi bạn chỉnh sửa code, hãy chạy 2 lệnh sau trong Terminal để công cụ `ruff` tự động căn chỉnh định dạng và rà soát lỗi tiềm ẩn:

```
ruff format .
ruff check .

```

## PHẦN 5\. QUẢN LÝ PHIÊN BẢN VỚI `Git` VÀ ĐẨY KHO LÊN `GitHub`

Nếu bạn muốn lưu lịch sử từng bước chỉnh sửa trên máy local và đồng bộ an toàn lên GitHub (tránh tình trạng đặt tên `code_final_v2_sua_lai.py`), hãy thực hiện theo quy trình chuẩn sau trong Terminal của VS Code:

### Bước 5.1. Cấu hình Git một lần duy nhất trên máy (nếu chưa từng cấu hình)

```
git config --global user.name "Nguyen Duc Huy"
git config --global user.email "email_cua_ban@example.com"
git config --global init.defaultBranch main
git config --global core.editor "code --wait"

```

*(Lệnh cuối giúp Git luôn mở VS Code thay vì trình soạn thảo Vim dễ gây kẹt màn hình).*

### Bước 5.2. Khởi tạo kho Git tại `C:\Users\Admin\Desktop\CodespaceGNN` và tạo bản lưu (`commit`) đầu tiên

Dự án đã có sẵn tệp `.gitignore` chặn `.venv/` và `__pycache__/` để kho Git luôn gọn nhẹ. Trong Terminal đứng tại `C:\Users\Admin\Desktop\CodespaceGNN`, gõ:

```
git init
git status --short

```

* Bạn sẽ thấy các thư mục mã nguồn và dữ liệu có dấu `??`, nhưng **tuyệt đối không có `.venv/` hay `__pycache__/`** nhờ tệp `.gitignore`.  
    
* Đưa toàn bộ dự án vào vùng chờ (*staging area*) và chụp bản lưu đầu tiên:

```
git add .
git commit -m "Khoi tao cau truc chuan CodespaceGNN (Step 0 -> Step 4, tests, notebooks, figures)"
git log --oneline

```

### Bước 5.3. Đưa kho lên GitHub để sao lưu an toàn

1. Mở trình duyệt vào `https://github.com`, bấm **New repository**, đặt tên kho là `CodespaceGNN`, chọn **Private** (hoặc Public), và **KHÔNG tích chọn** thêm `README`, `.gitignore` hay `license` (để kho trống hoàn toàn).  
     
2. Sao chép địa chỉ HTTPS của kho vừa tạo và gõ 2 lệnh sau vào Terminal:

```
git remote add origin https://github.com/<ten-tai-khoan-github-cua-ban>/CodespaceGNN.git
git push -u origin main

```

Từ những buổi làm việc sau, hãy duy trì **4 thói quen vàng**:

1. Mở máy ra: gõ `git pull` trước khi sửa mã.  
     
2. Mỗi khi sửa xong một hàm và chạy thử `python -m pytest` thấy đúng: gõ `git status` \$\\to\$ `git add <tệp>` \$\\to\$ `git commit -m "Mô tả rõ việc vừa làm"`.  
     
3. Lỡ tay sửa hỏng một tệp chưa commit: gõ `git restore <tệp>` để quay về bản commit gần nhất.  
     
4. Trước khi đóng máy: gõ `git push` để đẩy bản mới nhất lên GitHub.
