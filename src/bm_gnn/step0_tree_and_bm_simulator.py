#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 0: Xây dựng Đồ thị Cây Phả hệ 295 Nút, Ma trận Đồng tổ tiên S, Phân rã Cholesky
        và Bộ Mô phỏng Xuôi Quá trình Chuyển động Brown (BM SDE).

Tham chiếu:
  - Hình K1: Không gian - Thời gian Quá trình Khuếch tán BM SDE & Ma trận S.
  - Hình K4: Ánh xạ Cây Phả hệ Newick sang Đồ thị Hai chiều PyTorch Geometric (588 cạnh).
  - Tiểu luận Mục 2.1 - 2.3 & Cell 1 (dòng 26-238) trong Master Notebook.
"""

# 1. Thư viện chuẩn của Python
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 2. Thư viện bên thứ ba
import numpy as np
import pandas as pd
from scipy import linalg

# 3. Mã của chính dự án
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_gnn.config_paths import (
    META_PATH,
    NWK_PATH,
    PROCESSED_DATA_DIR,
    ensure_output_directories,
)


def c4_factor(df: int) -> float:
    """
    Tính hằng số khử chệch căn bậc hai c_4(df) của phân phối Chi tại bậc tự do df:
        E[sqrt(chi^2_df / df)] = c_4(df) = sqrt(2 / df) * Gamma((df + 1) / 2) / Gamma(df / 2).
    Tại df = 147: c_4(147) = 0.99830101 (và c_4(147)^(-1) = 1.00170188).
    """
    if df <= 0:
        raise ValueError("Bậc tự do df phải là số nguyên dương.")
    return math.sqrt(2.0 / df) * math.exp(
        math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0)
    )


def compute_reusable_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_scale: Optional[np.ndarray] = None,
    pred_se: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Tính bộ 11 chỉ số đánh giá thống kê chuẩn hóa cho ước lượng tham số:
      - MSE, RMSE, MAE, MedianAE, P90_AE, P95_AE, Bias
      - MAPE_pct (%), R2, Pearson_r
      - NRMSE_by_sigma (chuẩn hóa theo sigma_k để đối chiếu ngưỡng Cramér-Rao sqrt(v_rho))
      - PICP_95_pct (độ phủ thực nghiệm của khoảng dự đoán tin cậy 95%).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    err = y_pred_arr - y_true_arr
    abs_err = np.abs(err)

    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_err))
    med_ae = float(np.median(abs_err))
    p90_ae = float(np.percentile(abs_err, 90))
    p95_ae = float(np.percentile(abs_err, 95))
    bias = float(np.mean(err))
    mape = float(np.mean(abs_err / np.maximum(np.abs(y_true_arr), 1e-8)) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-15))
    corr = float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1]) if len(y_true_arr) > 1 else 1.0

    out: Dict[str, float] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MedianAE": med_ae,
        "P90_AE": p90_ae,
        "P95_AE": p95_ae,
        "Bias": bias,
        "MAPE_pct": mape,
        "R2": r2,
        "Pearson_r": corr,
    }
    if sigma_scale is not None:
        sig_arr = np.maximum(np.asarray(sigma_scale, dtype=np.float64), 1e-8)
        mse_norm = float(np.mean((err / sig_arr) ** 2))
        out["MSE_norm_by_sigma2"] = mse_norm
        out["NRMSE_by_sigma"] = float(math.sqrt(mse_norm))
    if pred_se is not None:
        se_arr = np.asarray(pred_se, dtype=np.float64)
        out["PICP_95_pct"] = float(np.mean(abs_err <= 1.96 * se_arr) * 100.0)
    return out


class PhyloTreeNode:
    """Biểu diễn một đỉnh (nút lá hoặc nút phân nhánh nội bộ) trên cây phả hệ định thời."""

    def __init__(
        self,
        name: Optional[str] = None,
        length: float = 0.0,
        parent: Optional["PhyloTreeNode"] = None,
    ) -> None:
        self.name: Optional[str] = name
        self.length: float = float(length) if length is not None else 0.0
        self.parent: Optional["PhyloTreeNode"] = parent
        self.children: List["PhyloTreeNode"] = []
        self.topo_depth: int = 0
        self.dist_from_root: float = 0.0
        self.subtree_leaves: int = 0
        self.node_id: int = -1
        self.y_pos: float = 0.0

    def is_leaf(self) -> bool:
        """Trả về True nếu nút không có nhánh con (nút lá hiện sinh)."""
        return len(self.children) == 0


class FromScratchPhylogenyBuilder:
    """
    Trích xuất cấu trúc toán học và đồ thị PyTorch Geometric từ chuỗi Newick:
      1. Đánh chỉ số |V| = 295 nút: nút lá 0..147, nút nội bộ hậu tố 148..294 (gốc rho = 294).
      2. Dựng 588 cạnh có hướng hai chiều (`edge_index` (2, 588) & `edge_attr` (588, 4)).
      3. Dựng tensor đặc trưng nút 6 kênh `X_k` (295, 6) với Zero-Masking tại 147 nút nội bộ.
      4. Tính ma trận đồng tổ tiên S (148 x 148), nhân tử Cholesky L_S, vec11, v_root và w_GLS.
    """

    def __init__(
        self,
        nwk_path: Path = NWK_PATH,
        meta_path: Path = META_PATH,
    ) -> None:
        self.nwk_path = Path(nwk_path)
        self.meta_path = Path(meta_path)
        self.nwk_str = self.nwk_path.read_text(encoding="utf-8").strip()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.species_names: List[str] = self.meta.get(
            "leaf_names", self.meta.get("taxa_names")
        )
        self.n_leaves: int = len(self.species_names)
        self.leaf_to_col: Dict[str, int] = {
            name: i for i, name in enumerate(self.species_names)
        }

        (
            self.root,
            self.postorder_nodes,
            self.all_nodes,
            self.nodes_df,
            self.edges_df,
        ) = self._parse_newick_and_index_nodes()
        self.graph_tensors: Dict[str, np.ndarray] = self._build_pyg_graph_topology()
        (
            self.S_mat,
            self.L_S,
            self.y_1,
            self.vec11,
            self.v_root,
            self.w_gls,
            self.s_bar_off,
            self.kappa_tree,
        ) = self._build_covariance_and_cholesky()

    def _parse_newick_and_index_nodes(
        self,
    ) -> Tuple[
        PhyloTreeNode,
        List[PhyloTreeNode],
        List[PhyloTreeNode],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        s = self.nwk_str[:-1].strip() if self.nwk_str.endswith(";") else self.nwk_str
        tokens = re.findall(r"\(|\)|,|[^(),:;]+|:[^(),:;]+", s)
        root = PhyloTreeNode(name="Root", length=0.0)
        curr = root
        for tok in tokens:
            if tok == "(":
                child = PhyloTreeNode(parent=curr)
                curr.children.append(child)
                curr = child
            elif tok == ",":
                parent = curr.parent
                assert parent is not None
                child = PhyloTreeNode(parent=parent)
                parent.children.append(child)
                curr = child
            elif tok == ")":
                assert curr.parent is not None
                curr = curr.parent
            elif tok.startswith(":"):
                curr.length = float(tok[1:])
            else:
                curr.name = tok.strip()

        postorder: List[PhyloTreeNode] = []
        leaf_order: List[PhyloTreeNode] = []

        def dfs(u: PhyloTreeNode, depth: int, dist: float) -> None:
            u.topo_depth = depth
            u.dist_from_root = dist
            if u.is_leaf():
                u.subtree_leaves = 1
                u.y_pos = float(len(leaf_order))
                leaf_order.append(u)
            else:
                u.subtree_leaves = 0
                for ch in u.children:
                    dfs(ch, depth + 1, dist + ch.length)
                    u.subtree_leaves += ch.subtree_leaves
                u.y_pos = float(np.mean([ch.y_pos for ch in u.children]))
            postorder.append(u)

        dfs(root, 0, 0.0)

        int_id = self.n_leaves
        for u in postorder:
            if u.is_leaf():
                assert u.name in self.leaf_to_col, f"Không tìm thấy loài {u.name}"
                u.node_id = self.leaf_to_col[u.name]
            else:
                u.node_id = int_id
                if not u.name:
                    u.name = f"Internal_{int_id}"
                int_id += 1

        all_nodes = sorted(postorder, key=lambda node: node.node_id)

        node_rows = []
        edge_rows = []
        for u in all_nodes:
            ntype = (
                "leaf"
                if u.is_leaf()
                else ("root" if u.parent is None else "internal")
            )
            node_rows.append(
                {
                    "node_id": u.node_id,
                    "name": u.name,
                    "node_type": ntype,
                    "is_leaf": int(u.is_leaf()),
                    "is_root": int(u.parent is None),
                    "topo_depth": u.topo_depth,
                    "dist_from_root": u.dist_from_root,
                    "branch_length_to_parent": u.length,
                    "subtree_leaves": u.subtree_leaves,
                }
            )
            for ch in u.children:
                dt = max(ch.length, 1e-8)
                # Cạnh đi lên (bottom-up: con -> cha)
                edge_rows.append(
                    {
                        "src": ch.node_id,
                        "dst": u.node_id,
                        "branch_length": ch.length,
                        "log_dt": math.log(dt),
                        "inv_dt": 1.0 / (dt + 1e-4),
                        "direction": 1,
                    }
                )
                # Cạnh đi xuống (top-down: cha -> con)
                edge_rows.append(
                    {
                        "src": u.node_id,
                        "dst": ch.node_id,
                        "branch_length": ch.length,
                        "log_dt": math.log(dt),
                        "inv_dt": 1.0 / (dt + 1e-4),
                        "direction": -1,
                    }
                )

        return (
            root,
            postorder,
            all_nodes,
            pd.DataFrame(node_rows),
            pd.DataFrame(edge_rows),
        )

    def _build_pyg_graph_topology(self) -> Dict[str, np.ndarray]:
        src_list, dst_list, attr_list = [], [], []
        left_ids, right_ids, dt_left, dt_right = [], [], [], []
        for u in self.all_nodes:
            if not u.is_leaf():
                ch_l, ch_r = u.children[0], u.children[1]
                left_ids.append(ch_l.node_id)
                right_ids.append(ch_r.node_id)
                dt_left.append(ch_l.length)
                dt_right.append(ch_r.length)
                for ch in (ch_l, ch_r):
                    dt = max(ch.length, 1e-8)
                    src_list.append(ch.node_id)
                    dst_list.append(u.node_id)
                    attr_list.append([dt, math.log(dt), 1.0 / (dt + 1e-4), +1.0])
                    src_list.append(u.node_id)
                    dst_list.append(ch.node_id)
                    attr_list.append([dt, math.log(dt), 1.0 / (dt + 1e-4), -1.0])

        return {
            "edge_index": np.array([src_list, dst_list], dtype=np.int64),
            "edge_attr": np.array(attr_list, dtype=np.float64),
            "left_ids": np.array(left_ids, dtype=np.int64),
            "right_ids": np.array(right_ids, dtype=np.int64),
            "dt_left": np.array(dt_left, dtype=np.float64),
            "dt_right": np.array(dt_right, dtype=np.float64),
        }

    def build_pyg_node_feature_tensor_X(self, x_leaf_row: np.ndarray) -> np.ndarray:
        """
        Dựng ma trận đặc trưng nút X_k thuộc R^(295 x 6):
          [x_obs * m_leaf, m_leaf, m_root, dist_from_root, branch_length, subtree_leaf_frac].
        """
        num_nodes = len(self.all_nodes)
        x_tensor = np.zeros((num_nodes, 6), dtype=np.float64)
        for u in self.all_nodes:
            nid = u.node_id
            is_lf = 1.0 if u.is_leaf() else 0.0
            is_rt = 1.0 if u.parent is None else 0.0
            val = float(x_leaf_row[nid]) if u.is_leaf() else 0.0
            x_tensor[nid, 0] = val
            x_tensor[nid, 1] = is_lf
            x_tensor[nid, 2] = is_rt
            x_tensor[nid, 3] = u.dist_from_root
            x_tensor[nid, 4] = u.length
            x_tensor[nid, 5] = u.subtree_leaves / float(self.n_leaves)
        return x_tensor

    def _build_covariance_and_cholesky(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray,
        float,
        float,
    ]:
        # Tính ma trận đồng tổ tiên S_ij = dist(rho, MRCA(i, j)) bằng quy hoạch động trên cây
        leaf_nodes_map: Dict[int, List[int]] = {}
        for u in self.postorder_nodes:
            if u.is_leaf():
                leaf_nodes_map[u.node_id] = [u.node_id]
            else:
                leaves_u: List[int] = []
                for ch in u.children:
                    leaves_u.extend(leaf_nodes_map[ch.node_id])
                leaf_nodes_map[u.node_id] = leaves_u

        s_mat = np.zeros((self.n_leaves, self.n_leaves), dtype=np.float64)
        for u in self.postorder_nodes:
            if u.is_leaf():
                s_mat[u.node_id, u.node_id] = u.dist_from_root
            else:
                ch_l, ch_r = u.children[0], u.children[1]
                for i_lf in leaf_nodes_map[ch_l.node_id]:
                    for j_lf in leaf_nodes_map[ch_r.node_id]:
                        s_mat[i_lf, j_lf] = u.dist_from_root
                        s_mat[j_lf, i_lf] = u.dist_from_root

        l_s = linalg.cholesky(s_mat, lower=True)
        ones = np.ones(self.n_leaves, dtype=np.float64)
        y_1 = linalg.solve_triangular(l_s, ones, lower=True)
        vec11 = float(np.dot(y_1, y_1))
        v_root = 1.0 / vec11
        w_gls = linalg.cho_solve((l_s, True), ones) / vec11

        off_mask = ~np.eye(self.n_leaves, dtype=bool)
        s_bar_off = float(np.mean(s_mat[off_mask]))
        kappa_tree = float(np.mean(np.diag(s_mat)) - s_bar_off)
        return s_mat, l_s, y_1, vec11, v_root, w_gls, s_bar_off, kappa_tree

    def simulate_forward_bm_preorder(
        self,
        mu: float,
        sigma: float,
        n_replicates: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Mô phỏng xuôi quá trình Chuyển động Brown (BM SDE) từ gốc xuống 148 lá
        bằng phép duyệt tiền tố (Preorder Traversal: Gốc -> Nội bộ -> Lá).
        Trả về ma trận X_leaves có kích thước (n_replicates, 148).
        """
        if rng is None:
            rng = np.random.default_rng(42)
        node_vals = np.zeros((n_replicates, len(self.all_nodes)), dtype=np.float64)
        # Duyệt ngược danh sách hậu tố chính là duyệt tiền tố (từ gốc 294 xuống lá 0..147)
        for u in reversed(self.postorder_nodes):
            if u.parent is None:
                node_vals[:, u.node_id] = mu
            else:
                parent_val = node_vals[:, u.parent.node_id]
                d_x = rng.normal(
                    loc=0.0,
                    scale=sigma * math.sqrt(max(u.length, 0.0)),
                    size=n_replicates,
                )
                node_vals[:, u.node_id] = parent_val + d_x
        return node_vals[:, : self.n_leaves]


def run_step0_verification() -> Dict[str, float]:
    """Thực thi kiểm định Bước 0 và lưu các bảng cấu trúc đồ thị vào `data/processed/`."""
    ensure_output_directories()
    builder = FromScratchPhylogenyBuilder()

    # Lưu ma trận S, S^-1, w_GLS và danh sách nút/cạnh đồ thị
    s_df = pd.DataFrame(
        builder.S_mat, index=builder.species_names, columns=builder.species_names
    )
    s_df.to_csv(PROCESSED_DATA_DIR / "ants_phylo_structure_matrix_S.csv")

    s_inv = linalg.cho_solve(
        (builder.L_S, True), np.eye(builder.n_leaves, dtype=np.float64)
    )
    inv_df = pd.DataFrame(
        s_inv, index=builder.species_names, columns=builder.species_names
    )
    inv_df["w_GLS"] = builder.w_gls
    inv_df.to_csv(PROCESSED_DATA_DIR / "ants_phylo_inverse_and_gls_weights.csv")

    builder.nodes_df.to_csv(
        PROCESSED_DATA_DIR / "step3_ant_phylogeny_graph_nodes.csv", index=False
    )
    builder.edges_df.to_csv(
        PROCESSED_DATA_DIR / "step3_ant_phylogeny_graph_edges.csv", index=False
    )

    logdet_s = float(2.0 * np.sum(np.log(np.diag(builder.L_S))))
    summary = {
        "n_leaves": builder.n_leaves,
        "n_internal_nodes": len(builder.all_nodes) - builder.n_leaves,
        "n_total_nodes": len(builder.all_nodes),
        "n_directed_edges": len(builder.edges_df),
        "logdet_S": logdet_s,
        "vec11": builder.vec11,
        "v_root": builder.v_root,
        "crlb_std_factor": math.sqrt(builder.v_root),
        "s_bar_off": builder.s_bar_off,
        "kappa_tree": builder.kappa_tree,
        "c4_147": c4_factor(147),
        "w_gls_max_outgroup": float(np.max(builder.w_gls)),
        "w_gls_min_crown": float(np.min(builder.w_gls)),
    }
    return summary


if __name__ == "__main__":
    info = run_step0_verification()
    print("===== KẾT QUẢ KIỂM ĐỊNH BƯỚC 0: ĐỒ THỊ CÂY PHẢ HỆ & MA TRẬN ĐỒNG TỔ TIÊN S =====")
    print(f"  * Số loài lá hiện sinh (n)          : {info['n_leaves']}")
    print(f"  * Số nút nội bộ (n - 1)             : {info['n_internal_nodes']}")
    print(f"  * Tổng số nút đồ thị (|V| = 2n - 1) : {info['n_total_nodes']}")
    print(f"  * Tổng số cạnh có hướng (|E_dir|)   : {info['n_directed_edges']}")
    print(f"  * Định thức logarit ln det(S)       : {info['logdet_S']:.6f}")
    print(f"  * Tổng độ chính xác gốc vec11       : {info['vec11']:.6f}")
    print(f"  * Phương sai gốc đơn vị v_rho       : {info['v_root']:.6f} (sqrt(v_rho) = {info['crlb_std_factor']:.6f})")
    print(f"  * Hệ số suy giảm phương sai kappa_T : {info['kappa_tree']:.6f} (s_bar_off = {info['s_bar_off']:.6f})")
    print(f"  * Hằng số khử chệch c_4(147)        : {info['c4_147']:.8f}")
    print(f"  * Trọng số lớn nhất (Philanthus sp.): {info['w_gls_max_outgroup']:.6f} ({info['w_gls_max_outgroup']*100:.2f}%)")
    print(f"  * Trọng số nhỏ nhất (Nhánh ngọn)    : {info['w_gls_min_crown']:.6f} ({info['w_gls_min_crown']*100:.4f}%)")
