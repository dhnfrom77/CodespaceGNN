# DANH SÁCH VÀ MÔ TẢ KỸ THUẬT CHI TIẾT CÁC HÌNH MINH HỌA PHƯƠNG PHÁP LUẬN VÀ KIẾN TRÚC MẠNG NƠ-RON ĐỒ THỊ (GNNs) TRONG TIỂU LUẬN NGHIÊN CỨU KHOA HỌC

**Tài liệu đi kèm:** `Tieu_Luan_Nghien_Cuu_Khoa_Hoc_Tong_Hop_Uoc_Luong_Dong_Thoi_BM_GNN_VI.pdf`  
**Đề tài:** *Ước Lượng Đồng Thời Hai Tham Số Của Mô Hình Chuyển Động Brown Trên Cây Phả Hệ Họ Kiến (Formicidae, $n = 148$ Loài) Bằng Mạng Nơ-Ron Đồ Thị Đẳng Biến Hai Luồng Và Suy Luận Hậu Nghiệm Bayes Thực Nghiệm Khấu Hao*

Trong tiểu luận nghiên cứu khoa học tổng hợp (`Tieu_Luan_Nghien_Cuu_Khoa_Hoc_Tong_Hop_Uoc_Luong_Dong_Thoi_BM_GNN_VI.pdf`), 7 biểu đồ thực nghiệm độ phân giải cao (bao gồm **Biểu đồ Cây Phả hệ Định thời Siêu dài 148 Loài Kiến**, **Sơ đồ Kiến trúc Tổng thể 6 Mô hình GNN $\mathcal{M}_1 \to \mathcal{M}_{5B}$**, và **5 Tổ hợp Biểu đồ Phân tán Đa quy mô**) đã được lập trình và nhúng trực tiếp vào bài viết. Bên cạnh đó, đối với **8 sơ đồ vi mạch kiến trúc học sâu và hình học phương pháp luận chuyên sâu** đòi hỏi thiết kế bằng công cụ đồ họa vector chuyên dụng (TikZ/PGFPlots, Adobe Illustrator, hoặc BioRender) khi xuất bản trên tạp chí quốc tế, tài liệu này cung cấp danh mục tên file chuẩn và đặc tả kỹ thuật — thị giác chi tiết đến từng khối tensor, ký hiệu toán học, bảng màu và luồng mũi tên.

---

## Tổng Hợp Danh Mục 8 Hình Minh Họa Phương Pháp Luận & Kiến Trúc GNN (`Hình K1` $\to$ `Hình K8`)

| Mã hình | Tên file chuẩn khuyến nghị (`.pdf` / `.svg` / `.png`) | Vị trí chèn trong Tiểu luận | Nội dung cốt lõi & Tài liệu tham khảo gốc |
| :--- | :--- | :--- | :--- |
| **Hình K1** | `fig_method_01_spatiotemporal_bm_sde_and_mrca_covariance_3d.pdf` | Mục 2.2 (Cấu trúc Cây & SDE) | Biểu đồ 3D quá trình khuếch tán SDE $\mathrm{d}X(t) = \sigma\,\mathrm{d}W(t)$ dọc theo nhánh cây và hình học ma trận đồng tổ tiên $\mathbf{S}$ ([1], [2], [4], [33]). |
| **Hình K2** | `fig_method_02_hilbert_space_orthogonal_decomposition_147_plus_1_dof.pdf` | Mục 3.1 (Định lý Phân rã Trực giao) | Hình học không gian tích trong $\langle \mathbf{u},\mathbf{v}\rangle_{\mathbf{S}^{-1}}$ tách $\mathbb{R}^{148} = \mathcal{V}_{\mathrm{root}} (1\text{ DOF}) \oplus \mathcal{V}_{\mathrm{int}} (147\text{ DOF})$ ([1], [3], [4]). |
| **Hình K3** | `fig_method_03_phylolm_linear_time_three_point_pruning_flowchart.pdf` | Mục 3.2 (Tiêu chuẩn Vàng `phylolm`) | Lưu đồ thuật toán cắt tỉa cấu trúc 3 điểm tuyến tính $\mathcal{O}(n)$ của gói `phylolm` (Ho & Ané, 2014) ([3], [5]). |
| **Hình K4** | `fig_gnn_04_phylogenetic_tree_to_pyg_bidirected_graph_tensors.pdf` | Mục 4.1 (Biểu diễn Đồ thị PyG) | Sơ đồ ánh xạ cây Newick 148 loài sang đồ thị hai chiều `PyTorch Geometric` ($|V|=295$ nút, $|E|=588$ cạnh) ([8], [14], [15]). |
| **Hình K5** | `fig_gnn_05_six_tier_architectural_ablation_matrix_m1_to_m5b.pdf` | Mục 4.2 (Phân cấp $\mathcal{M}_1 \to \mathcal{M}_{5B}$) | Ma trận đối sánh cơ chế truyền thông điệp của 6 kiến trúc GNN từ `M1_DeepSets` đến `M5B_BayesMMSE` ([6]–[15]). |
| **Hình K6** | `fig_gnn_06_directed_bottom_up_tree_mpnn_cell_microarchitecture.pdf` | Mục 4.2 (Tầng `DirectedBottomUpTreeMPNNLayer`) | Sơ đồ vi mạch nội bộ của tế bào truyền thông điệp cây có hướng tại nút phân nhánh $u \in [148, 294]$ ([8], [15]). |
| **Hình K7** | `fig_gnn_07_m5a_dual_stream_equivariant_phylognn_and_nll_loss.pdf` | Mục 4.2 (Kiến trúc `PhyloGNN_M5A`) | Kiến trúc hai luồng Đẳng biến/Bất biến tịnh tiến của `M5A` và cơ chế cân bằng đạo hàm dị phương sai $\hat{\sigma}_k^{-2}$ ([15]). |
| **Hình K8** | `fig_gnn_08_m5b_amortized_empirical_bayes_shrinkage_and_cohort_updating.pdf` | Mục 4.2 & Mục 7.1 (Kiến trúc `PhyloGNN_M5B` & Hậu-BM) | Kiến trúc đầu đọc co rút Chuẩn Cụt 1D trên $[-5, 5]$, lưới tích phân Log-Normal 250 điểm, cập nhật đoàn hệ $N$ cây và mở rộng hậu-BM ([20]–[30]). |

---

## Đặc Tả Kỹ Thuật Và Thị Giác Chi Tiết Cho Từng Hình

### 1. Hình K1: `fig_method_01_spatiotemporal_bm_sde_and_mrca_covariance_3d.pdf`
* **Tiêu đề đầy đủ:** *Không gian — Thời gian của Quá trình Khuếch tán Chuyển động Brown $\mathrm{d}X(t) = \sigma\,\mathrm{d}W(t)$ trên Cây Phả hệ Định thời và Cơ chế Hình thành Ma trận Hiệp phương sai Đồng tổ tiên $\mathbf{S} \in \mathbb{R}^{148 \times 148}$.*
* **Bố cục tổng thể (3 khung ngang A – B – C, tỷ lệ 16:7):**
  1. **Khung (A) — Phối cảnh 3D Quỹ đạo Ngẫu nhiên trên Cây Phân nhánh:**
     * Trục hoành $t \in [0, 1.0]$ biểu diễn thời gian tiến hóa chuẩn hóa từ gốc tổ tiên $\rho$ ($t=0$) đến hiện tại ($t=1.0$); trục sâu $y$ biểu diễn vị trí hình học của các nhánh loài kiến; trục đứng $z = X(t)$ biểu diễn giá trị tính trạng liên tục khởi phát từ $X(\rho) = \mu$.
     * Vẽ một cây con đại diện gồm 5 loài kiến (ví dụ: *Eutetramorium sp.*, *Colobostruma unicorna*, *Vollenhovia sp.*, *Martialis heureka*, và loài ngoại nhóm tách sớm *Philanthus sp.* tại $t=0$).
     * Trên mỗi nhánh $e = (u, v)$ có độ dài $\Delta t_e$, vẽ quỹ đạo chuyển động Brown liên tục (đường gấp khúc ngẫu nhiên mảnh) kèm ống mật độ xác suất Gauss 95% mở rộng theo căn bậc hai thời gian $\pm 1.96\,\sigma\sqrt{\Delta t_e}$.
     * Tại nút phân nhánh nội bộ $u = \mathrm{MRCA}(i, j)$ ở thời điểm $t_u = s_{ij}$, đánh dấu điểm rẽ nhánh màu đỏ tươi: trước $t_u$, hai loài $i$ và $j$ đi chung một quỹ đạo duy nhất (tạo nên hiệp phương sai $\mathrm{Cov}(X_i, X_j) = \sigma^2 s_{ij}$); sau $t_u$, hai nhánh tách làm hai quỹ đạo Wiener độc lập $\Delta W_{u \to i} \perp \Delta W_{u \to j}$.
  2. **Khung (B) — Cấu trúc Khối Phân cấp của Ma trận $\mathbf{S} = [s_{ij}]_{148 \times 148}$:**
     * Hiển thị bản đồ nhiệt (heatmap) của ma trận $\mathbf{S}$ kích thước $148 \times 148$ được sắp xếp theo thứ tự phân nhánh của cây Newick.
     * Chú thích đường chéo chính $s_{ii} = T_i \approx 1.0000$, các khối vuông đồng tổ tiên dọc đường chéo tương ứng với các phân họ kiến lớn (*Myrmicinae*, *Formicinae*, *Ponerinae*, *Dorylinae*), và hàng/cột cuối cùng số 148 của loài ngoại nhóm *Philanthus sp.* có $s_{i,148} = 0.0$ hoàn toàn tách biệt.
  3. **Khung (C) — Hiệu ứng Co Phương sai Mẫu $\kappa_{\mathcal{T}} = 1 - \bar{s}_{\mathrm{off}} = 0.615125$:**
     * So sánh 2 biểu đồ mật độ phương sai mẫu giữa các loài $s_{\mathrm{sample}}^2(\mathbf{x}_k) = \frac{1}{147}\sum_{i=1}^{148}(x_{k,i} - \bar{x}_k)^2$: đường nét đứt giả định độc lập hình sao ($\mathbb{E}[s_{\mathrm{sample}}^2] = \sigma^2$) và đường liền nét trên cây kiến thực tế ($\mathbb{E}[s_{\mathrm{sample}}^2] = \kappa_{\mathcal{T}}\sigma^2 = 0.615125\,\sigma^2$ do tỷ trọng đồng tổ tiên trung bình $\bar{s}_{\mathrm{off}} = 0.384875$).

### 2. Hình K2: `fig_method_02_hilbert_space_orthogonal_decomposition_147_plus_1_dof.pdf`
* **Tiêu đề đầy đủ:** *Hình học Không gian Tích trong $\langle \mathbf{u}, \mathbf{v} \rangle_{\mathbf{S}^{-1}}$ và Định lý Phân rã Trực giao Thống kê Đủ Giải Độc lập Bài toán Ước lượng Đồng thời $(\mu, \sigma) \mid \mathbf{x}$.*
* **Bố cục tổng thể (2 khung lớn A – B):**
  1. **Khung (A) — Phép Chiếu Trực giao $\mathbf{P}_{\mathrm{root}}$ trong Không gian $\mathbb{R}^{148}$ trang bị Metric $\mathbf{S}^{-1}$:**
     * Vẽ siêu phẳng 147 chiều $\mathcal{V}_{\mathrm{int}} = \{\mathbf{r} \in \mathbb{R}^{148} : \mathbf{1}_{148}^\top \mathbf{S}^{-1}\mathbf{r} = 0\}$ nằm ngang (màu xanh lam nhạt) và trục 1 chiều $\mathcal{V}_{\mathrm{root}} = \mathrm{span}(\mathbf{1}_{148})$ vuông góc theo tích trong $\mathbf{S}^{-1}$ (màu đỏ sẫm).
     * Biểu diễn vector quan sát đã dịch tâm thực $\mathbf{x} - \mu\mathbf{1}_{148}$ là cạnh huyền của một tam giác vuông chính xác trong metric $\mathbf{S}^{-1}$, phân rã thành hai vector thành phần vuông góc:
       $$\mathbf{x} - \mu\mathbf{1}_{148} = \underbrace{(\mathbf{x} - \hat{\mu}_{\mathrm{GLS}}\mathbf{1}_{148})}_{\mathbf{r}_{\mathrm{int}} \in \mathcal{V}_{\mathrm{int}}\; (\mathrm{df}=147)} + \underbrace{(\hat{\mu}_{\mathrm{GLS}} - \mu)\mathbf{1}_{148}}_{\mathbf{r}_{\mathrm{root}} \in \mathcal{V}_{\mathrm{root}}\; (\mathrm{df}=1)}.$$
     * Ghi rõ định lý Pythagoras tổng quát: $Q_0(\mathbf{x},\mu) = Q_{\mathrm{internal}}(\mathbf{x}) + \mathrm{vec11}\cdot(\hat{\mu}_{\mathrm{GLS}} - \mu)^2$, với sai số số học kiểm chứng $\max|\Delta Q| = 1.819 \times 10^{-12}$.
  2. **Khung (B) — Sự Độc lập Thống kê Hoàn toàn Giữa Hai Luồng Suy luận:**
     * Vẽ sơ đồ tách đôi thống kê đủ: nhánh trên chiếu lên $\mathcal{V}_{\mathrm{root}}$ thu được $\hat{\mu}_{\mathrm{GLS}} = \mathbf{w}_{\mathrm{GLS}}^\top \mathbf{x} \sim \mathcal{N}(\mu, \sigma^2 v_{\rho})$ không chứa tham số $\sigma$ trong trọng số; nhánh dưới chiếu lên $\mathcal{V}_{\mathrm{int}}$ thu được 147 tương phản trực giao $C_1, \dots, C_{147} \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$ bất biến hoàn toàn với $\mu$.
     * Đóng khung kết luận phương pháp luận: *Bài toán ước lượng đồng thời $(\mu, \sigma) \mid \mathbf{x}$ được giải trực tiếp trong một bước duy nhất mà không cần giải bài toán phụ 1 tham số.*

### 3. Hình K3: `fig_method_03_phylolm_linear_time_three_point_pruning_flowchart.pdf`
* **Tiêu đề đầy đủ:** *Lưu đồ Thuật toán Cắt tỉa Cấu trúc 3 Điểm Tuyến tính $\mathcal{O}(n)$ của Tiêu chuẩn Vàng Cổ điển `phylolm` (Ho & Ané, 2014 [3]) và Cơ chế Biến đổi Nhánh Tổng quát.*
* **Bố cục tổng thể (3 tầng từ dưới lên trên):**
  1. **Tầng Đáy — Khởi tạo tại $n = 148$ Nút Lá ($v \in V_{\mathrm{leaf}}$):**
     * Hiển thị các khối nút lá $v = 0, \dots, 147$ với độ dài nhánh nối lên cha $t_v = \Delta t_v$.
     * Bên trong mỗi khối nút lá ghi bộ 6 thanh ghi trạng thái khởi tạo $\mathcal{O}(1)$:
       $$\mathrm{logd}_v = \ln t_v,\quad u_{1,v} = \frac{1}{t_v},\quad u_{X,v} = \frac{x_v}{t_v},\quad \mathrm{vec11}_v = \frac{1}{t_v},\quad P_{1,v} = \frac{x_v}{t_v},\quad PP_v = \frac{x_v^2}{t_v}.$$
  2. **Tầng Giữa — Phép Hợp nhất Đệ quy Hậu thứ tự tại $147$ Nút Nội bộ ($u \in V_{\mathrm{int}}$):**
     * Vẽ sơ đồ khối tại nút nội bộ $u$ nhận hai luồng thanh ghi từ con trái $l(u)$ và con phải $r(u)$, tính tổng trung gian $P_{\mathrm{ch}} = \mathrm{vec11}_l + \mathrm{vec11}_r$, $Y_{1,\mathrm{ch}} = u_{1,l} + u_{1,r}$, $Y_{X,\mathrm{ch}} = u_{X,l} + u_{X,r}$, hệ số nén nhánh $\mathrm{denom}_u = 1 + t_u P_{\mathrm{ch}}$, và cập nhật:
       $$\mathrm{vec11}_u = P_{\mathrm{ch}} - \frac{t_u Y_{1,\mathrm{ch}}^2}{\mathrm{denom}_u},\qquad P_{1,u} = P_{1,\mathrm{ch}} - \frac{t_u Y_{1,\mathrm{ch}} Y_{X,\mathrm{ch}}}{\mathrm{denom}_u},\qquad PP_u = PP_{\mathrm{ch}} - \frac{t_u Y_{X,\mathrm{ch}}^2}{\mathrm{denom}_u}.$$
  3. **Tầng Đỉnh — Đọc Kết quả tại Gốc $\rho$ (`id = 294`, $t_{\rho} = 0$) & Module Biến đổi Nhánh Đa dụng:**
     * Xuất trực tiếp $\hat{\mu}_{\mathrm{phylolm}} = P_{1,\rho}/\mathrm{vec11}_{\rho}$, $Q_{\mathrm{internal}} = PP_{\rho} - P_{1,\rho}^2/\mathrm{vec11}_{\rho}$, và $\hat{\sigma}_{\mathrm{REML}} = \sqrt{Q_{\mathrm{internal}}/147}$.
     * Khung bên phải minh họa lý do `phylolm` là Tiêu chuẩn Vàng: khi chuyển sang các mô hình OU ($\alpha$), Early Burst ($r$), hoặc Pagel ($\lambda, \kappa, \delta$), chỉ cần gắn một bộ biến đổi độ dài nhánh hiệu dụng $\Delta t_e \mapsto \Delta \tilde{t}_e(\boldsymbol{\theta})$ ở đầu vào mà giữ nguyên 100% lõi cắt tỉa $\mathcal{O}(n)$ [3, 5].

### 4. Hình K4: `fig_gnn_04_phylogenetic_tree_to_pyg_bidirected_graph_tensors.pdf`
* **Tiêu đề đầy đủ:** *Sơ đồ Ánh xạ Cây Phả hệ Định thời Newick sang Cấu trúc Đồ thị Hai chiều Có hướng trong `PyTorch Geometric` (`Data(x, edge_index, edge_attr)`).*
* **Bố cục tổng thể (2 khung đối xứng Trái – Phải):**
  1. **Khung Trái — Đánh số Nút Chuẩn hóa ($|V| = 295$) và Cặp Cạnh Hai chiều ($|E_{\mathrm{dir}}| = 588$):**
     * Minh họa một phân đoạn cây gồm các nút lá `0..147` (màu xanh lục) và các nút nội bộ hậu tố `148..294` (màu xanh đậm, gốc `294` màu đỏ).
     * Trên mỗi nhánh vật lý nối nút con $v$ và nút cha $u$, vẽ 2 mũi tên song song ngược chiều:
       * Mũi tên đi lên (màu xanh ngọc, `dir_flag = +1`): truyền thông điệp tích hợp độ chính xác từ lá lên gốc (`bottom_up_child_to_parent`).
       * Mũi tên đi xuống (màu cam nhạt, `dir_flag = -1`): truyền thông tin ngữ cảnh từ cha xuống con (`top_down_parent_to_child`).
  2. **Khung Phải — Cấu trúc Khối Tensor Đầu vào của Mạng GNN:**
     * Tensor chỉ số cạnh `edge_index` kích thước `(2, 588)` kiểu `torch.long`.
     * Tensor thuộc tính cạnh `edge_attr` kích thước `(588, 4)` kiểu `torch.float64`: $[\Delta t_e,\, (\Delta t_e + 10^{-3})^{-1},\, \ln(\Delta t_e),\, \mathrm{dir}_e]$.
     * Tensor đặc trưng nút `x` kích thước `(295, 6)`: $[x_{k,v}^{\mathrm{obs}},\, \delta_{k,v} = x_{k,v}-\bar{x}_k,\, m_v^{\mathrm{leaf}},\, r_v^{\mathrm{root}},\, T_v,\, d_{\mathrm{topo}}(v)/16]$, hiển thị rõ mặt nạ số $0$ tại 147 hàng nút nội bộ ($v \in [148, 294]$).

### 5. Hình K5: `fig_gnn_05_six_tier_architectural_ablation_matrix_m1_to_m5b.pdf`
* **Tiêu đề đầy đủ:** *Ma trận Trực quan Phân cấp 6 Kiến trúc Mạng Nơ-ron Đồ thị ($\mathcal{M}_1 \to \mathcal{M}_{5B}$): Từ Tập Điểm Mù Tô-pô đến Mạng Truyền Thông điệp Cây Đẳng biến Hai luồng và Co rút Bayes.*
* **Bố cục tổng thể (Lưới $2 \times 3$ gồm 6 ô kiến trúc đồng nhất):**
  * **Ô 1 (`M1_MLP_DeepSets` [6]):** Vẽ 148 nút lá rời rạc không có cạnh nối, đi qua mạng $\phi_{\boldsymbol{\theta}}$ và gộp trung bình/phân vị toàn cục $\to$ chú thích: *Mù cấu trúc cây ($R^2(\hat{\sigma}) = 0.5686$)*.
  * **Ô 2 (`M2_GCN_Unweighted` [7]):** Vẽ cây 295 nút với mọi cạnh có độ dày bằng nhau ($\Delta t_e \equiv 1$) và mũi tên hai chiều đối xứng $\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ $\to$ chú thích: *Mù độ dài nhánh thời gian ($R^2(\hat{\sigma}) = 0.8867$)*.
  * **Ô 3 (`M3_EdgeMPNN_GINE` [8, 9]):** Vẽ cây 295 nút có độ dày cạnh tỷ lệ với $1/\Delta t_e$ nhưng mũi tên truyền đẳng hướng cả hai chiều làm rò rỉ giá trị $0$ từ nút nội bộ xuống lá $\to$ chú thích: *Suy giảm biên độ đẳng hướng ($R^2(\hat{\sigma}) = 0.9790$)*.
  * **Ô 4 (`M4_PhyloGNN_RawMSE` [15]):** Vẽ mũi tên một chiều đi lên theo thứ tự hậu tố $148 \to 294$, nhưng hàm mất mát MSE không trọng số bị méo đạo hàm $2{,}500\times$ giữa $\sigma=0.1$ và $\sigma=5.0$ $\to$ chú thích: *Thiên lệch đạo hàm dị phương sai ($\mathrm{MSE}(\hat{\mu}) = 0.6318$)*.
  * **Ô 5 (`M5A_PhyloGNN_Unbiased_NLL` — Đề xuất 1):** Vẽ thân mạng truyền thông điệp hậu tố có cổng độ chính xác tách thành 2 luồng đầu đọc trực giao (Luồng 1 gộp bậc hai 147 tương phản $C_{k,u}^2$; Luồng 2 đọc nút gốc $H_{k,294}$) dưới hàm mất mát tự chuẩn hóa $\hat{\sigma}_k^{-2}$ $\to$ chú thích: *Hội tụ sát tuyệt đối `phylolm` ($R^2(\hat{\sigma}) = 0.996632$)*.
  * **Ô 6 (`M5B_PhyloGNN_BayesMMSE` — Đề xuất 2 [20, 21]):** Gắn thêm khối co rút hậu nghiệm Chuẩn Cụt $[-5,5]$ + lưới Log-Normal 250 điểm lên trên `M5A` $\to$ chú thích: *Giảm $29.74\%$ $\mathrm{MSE}(\hat{\mu})$ trên tập Test ($R^2(\hat{\mu}) = 0.9467$)*.

### 6. Hình K6: `fig_gnn_06_directed_bottom_up_tree_mpnn_cell_microarchitecture.pdf`
* **Tiêu đề đầy đủ:** *Sơ đồ Vi mạch Nội bộ (Micro-Architecture) của Tầng `DirectedBottomUpTreeMPNNLayer` tại Một Nút Phân nhánh Nội bộ $u \in \{148, \dots, 294\}$.*
* **Bố cục tổng thể (Sơ đồ mạch tính toán chi tiết từ 2 nút con $l(u), r(u)$ lên nút cha $u$):**
  1. **Đầu vào từ hai nhánh con:** Nhánh trái đưa vào cặp $(H_{k,l(u)}, v_{l(u)})$ và độ dài nhánh $\Delta t_{l(u)}$; nhánh phải đưa vào $(H_{k,r(u)}, v_{r(u)})$ và $\Delta t_{r(u)}$.
  2. **Khối Cổng Độ chính xác Khả vi (Learnable Branch Precision Gates):**
     * Hai tham số học được `log_gate_left[u]` ($g_{L,u}$) và `log_gate_right[u]` ($g_{R,u}$) đi qua hàm mũ $\exp(\cdot)$ rồi nhân với tổng phương sai tích lũy:
       $$d_L(u) = \left(v_{l(u)} + \Delta t_{l(u)}\right)\exp(g_{L,u}), \qquad d_R(u) = \left(v_{r(u)} + \Delta t_{r(u)}\right)\exp(g_{R,u}).$$
  3. **Khối Hợp nhất Trạng thái Nút (Precision-Weighted Convex Aggregation):**
     * Tính tổng phương sai nhánh $d_{\mathrm{sum}}(u) = d_L(u) + d_R(u)$ và hai trọng số chéo $w_L = d_R(u)/d_{\mathrm{sum}}(u)$, $w_R = d_L(u)/d_{\mathrm{sum}}(u)$ (bảo đảm $w_L + w_R \equiv 1$, giữ nguyên tính đẳng biến dịch chuyển).
     * Xuất trạng thái nút cha $H_{k,u} = w_L H_{k,l(u)} + w_R H_{k,r(u)}$ và phương sai cây con hiệu dụng $v_u = \frac{d_L(u)d_R(u)}{d_{\mathrm{sum}}(u)}$.
  4. **Khối Trích xuất Thông điệp Tương phản Chuẩn hóa (Standardized Contrast Message Extractor):**
     * Bộ trừ vi sai tính hiệu $H_{k,l(u)} - H_{k,r(u)}$ (triệt tiêu hoàn toàn hằng số dịch chuyển $\mu_k$) rồi chia cho $\sqrt{d_{\mathrm{sum}}(u)}$ để xuất thông điệp tương phản chuẩn hóa $C_{k,u} = \frac{H_{k,l(u)} - H_{k,r(u)}}{\sqrt{d_L(u)+d_R(u)}}$ đưa sang Luồng 1.

### 7. Hình K7: `fig_gnn_07_m5a_dual_stream_equivariant_phylognn_and_nll_loss.pdf`
* **Tiêu đề đầy đủ:** *Kiến trúc Chi tiết của Mô hình `PhyloGNN_M5A_UnbiasedNLL` và Cơ chế Cân bằng Đạo hàm Dị phương sai của Hàm mất mát `HeteroscedasticPhyloNLLLoss`.*
* **Bố cục tổng thể (2 phần: Sơ đồ luồng tensor bên trái & Đồ thị mặt phẳng đạo hàm bên phải):**
  1. **Phần Trái — Kiến trúc Hai Luồng Độc lập Thống kê:**
     * Từ tầng `DirectedBottomUpTreeMPNNLayer`, ma trận tương phản $\mathbf{C}_{\mathrm{int}} \in \mathbb{R}^{B \times 147}$ đi vào `QuadraticContrastScaleHead`: bình phương từng phần tử $\mathbf{C}_{\mathrm{int}}^{\odot 2}$, nhân vô hướng với vector trọng số đơn hình $\mathbf{w}_{\sigma} = \mathrm{Softmax}(\boldsymbol{\eta}_{\sigma}) \in \Delta^{146}$, lấy căn bậc hai và nhân với hệ số khử chệch khả vi $\gamma_{c_4}$ để xuất $\hat{\sigma}_{k,\mathcal{M}_{5A}}$.
     * Ma trận trạng thái nút $\mathbf{H}_{\mathrm{nodes}} \in \mathbb{R}^{B \times 295}$ đi vào `EquivariantRootLocationHead`: lấy cột gốc $\mathbf{H}_{:,294}$ cộng với tích vô hướng của 146 độ lệch nội bộ $(\mathbf{H}_{:,148:294} - \mathbf{H}_{:,294}\mathbf{1}_{146}^\top)\mathbf{w}_{\mu}$ để xuất $\hat{\mu}_{k,\mathcal{M}_{5A}}$.
  2. **Phần Phải — Cơ chế Tự Chuẩn hóa Đạo hàm bằng $\hat{\sigma}_k^{-2}$:**
     * Vẽ mũi tên phản hồi từ đầu ra $\hat{\sigma}_{k,\mathcal{M}_{5A}}^2$ sang mẫu số của hàm mất mát vị trí $\frac{1}{2}\left[\frac{(\hat{\mu}_k - \mu_k)^2}{\hat{\sigma}_k^2} + \ln\hat{\sigma}_k^2\right]$.
     * Đồ thị so sánh biên độ gradient theo $\sigma_k \in [0.1, 5.0]$: dưới hàm MSE thô ($\mathcal{M}_4$), tỷ trọng năng lượng gradient tại $\sigma=5.0$ lớn gấp $2{,}500$ lần tại $\sigma=0.1$; dưới hàm `HeteroscedasticPhyloNLLLoss` ($\mathcal{M}_{5A}$), đường tỷ trọng gradient phẳng tuyệt đối trên toàn dải $\sigma \in [0.1, 5.0]$.

### 8. Hình K8: `fig_gnn_08_m5b_amortized_empirical_bayes_shrinkage_and_cohort_updating.pdf`
* **Tiêu đề đầy đủ:** *Kiến trúc Chi tiết của Mô hình `PhyloGNN_M5B_BayesMMSE`: Đầu đọc Co rút Hậu nghiệm Chuẩn Cụt 1D, Lưới Tích phân Log-Normal 250 Điểm, Cập nhật Đoàn hệ $N$ Cây và Khả năng Mở rộng Hậu-BM.*
* **Bố cục tổng thể (3 khối chức năng liên hoàn A – B – C):**
  1. **Khối (A) — Cơ chế Co rút Hậu nghiệm Chuẩn Cụt (Truncated-Normal Posterior Mean) cho $\hat{\mu}$:**
     * Minh họa một đường cong mật độ Gauss $\mathcal{N}(\hat{\mu}_{\mathcal{M}_{5A}}, \hat{s}_{0,k}^2)$ khi $\hat{\mu}_{\mathcal{M}_{5A}} = -4.6$ và $\hat{s}_{0,k} = \hat{\sigma}_{\mathcal{M}_{5A}}\sqrt{v_{\rho}} = 1.5$: phần đuôi bên trái vượt ra ngoài biên $[-5, 5]$ bị cắt bỏ (tô màu xám gạch chéo), làm dịch chuyển kỳ vọng hậu nghiệm $\mathbb{E}[\mu \mid \mathbf{x}_k]$ vào phía trong miền $[-5, 5]$ một lượng chính xác bằng $\hat{s}_{0,k}\frac{\phi(\alpha_k)-\phi(\beta_k)}{\Phi(\beta_k)-\Phi(\alpha_k)}$.
     * Hiển thị nhánh song song `residual_shrinkage_mlp` (`Linear(4->64) -> LayerNorm -> GELU -> Linear(64->32) -> GELU -> Linear(32->2)`) nhận vector đặc trưng 4 chiều $[\hat{\mu}/5,\, \hat{\sigma}/5,\, \hat{s}_0,\, \tanh(\hat{\mu}/5)]$ để tinh chỉnh phi tuyến.
  2. **Khối (B) — Lưới Tích phân Hậu nghiệm Log-Normal 250 Điểm cho $\hat{\sigma}$ & Chế độ Đoàn hệ $N$ Cây:**
     * Minh họa phép nhân mật độ hợp lý Chi-bình phương 147 bậc tự do $p(Q_{\mathrm{int}} \mid \sigma)$ với tiên nghiệm Log-Normal thực nghiệm trên lưới 250 điểm $\sigma_g \in [0.1, 5.0]$ thông qua phép chuẩn hóa `Softmax` log-hậu nghiệm.
     * Hiển thị công tắc chuyển chế độ: khi suy luận trên đoàn hệ $N$ cây ($N \in \{100, \dots, 100{,}000\}$), độ lệch chuẩn nhiễu gốc giảm theo $\hat{s}_0^{(N)} = \hat{\sigma}_{\mathrm{group}}\sqrt{v_{\rho}/N} \to 0$, giúp ước lượng đoàn hệ hội tụ không chệch $\mathcal{O}(N^{-1/2})$.
  3. **Khối (C) — Khung Tái sử dụng Hoàn chỉnh cho Các Mô hình Tiến hóa Hậu-BM (OU, EB, Lévy):**
     * Vẽ sơ đồ cắm-và-chạy (Plug-and-Play SBI Architecture): chỉ cần thay khối sinh dữ liệu mô phỏng ở đầu vào (từ SDE Chuyển động Brown sang SDE Ornstein–Uhlenbeck đa chế độ [22–26] hoặc quá trình nhảy Lévy [28]), toàn bộ đồ thị PyG 295 nút, thân mạng `DirectedBottomUpTreeMPNNLayer` và đầu đọc hậu nghiệm của `M5B` được giữ nguyên để ước lượng đồng thời vector tham số tiến hóa tổng quát $\boldsymbol{\theta}$.
