#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 0: Xây dựng Đồ thị Cây Phả hệ 295 Nút, Ma trận Đồng tổ tiên S, Phân rã Cholesky,
        Bộ Mô phỏng Xuôi BM SDE và Khởi tạo Tập Test Mô phỏng Cố định (1,000 mẫu).
"""

# 1. Thư viện chuẩn của Python
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 2. Thư viện bên thứ ba
import numpy as np
import pandas as pd
from scipy import linalg

# 3. Mã của chính dự án
from bm_gnn.config_paths import (
    META_PATH,
    NUM_TEST_SAMPLES,
    NWK_PATH,
    PRIOR_MU_MAX,
    PRIOR_MU_MIN,
    PRIOR_SIGMA_MAX,
    PRIOR_SIGMA_MIN,
    PROCESSED_DATA_DIR,
    SEED_FIXED_TEST,
    TEST_SIM_CSV_PATH,
    ensure_output_directories,
)


def c4_factor(df: int) -> float:
    """Tính hằng số khử chệch căn bậc hai c_4(df) của phân phối Chi tại bậc tự do df."""
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
    """Tính bộ 11 chỉ số đánh giá thống kê chuẩn hóa cho ước lượng tham số."""
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
        return len(self.children) == 0


class FromScratchPhylogenyBuilder:
    """Trích xuất cấu trúc toán học và đồ thị PyTorch Geometric từ chuỗi Newick."""

    def __init__(
        self,
        nwk_path: Path = NWK_PATH,
        meta_path: Optional[Path] = META_PATH,
    ) -> None:
        self.nwk_path = Path(nwk_path)
        self.meta_path = Path(meta_path) if meta_path is not None else None
        self.nwk_str = self.nwk_path.read_text(encoding="utf-8").strip()
        s_clean = self.nwk_str[:-1].strip() if self.nwk_str.endswith(";") else self.nwk_str
        toks = re.findall(r"\(|\)|,|[^(),:;]+|:[^(),:;]+", s_clean)
        nwk_leaves = [
            tok.strip()
            for idx, tok in enumerate(toks)
            if tok not in ("(", ")", ",")
            and not tok.startswith(":")
            and (idx == 0 or toks[idx - 1] in ("(", ","))
        ]
        if self.meta_path is not None and self.meta_path.exists():
            loaded_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            cand_names = loaded_meta.get("leaf_names", loaded_meta.get("taxa_names", []))
            if set(cand_names) == set(nwk_leaves) and len(cand_names) == len(nwk_leaves):
                self.meta = loaded_meta
                self.species_names: List[str] = cand_names
            else:
                self.meta = {"leaf_names": nwk_leaves, "n_leaves": len(nwk_leaves), "source_file": self.nwk_path.name}
                self.species_names = nwk_leaves
        else:
            self.meta = {"leaf_names": nwk_leaves, "n_leaves": len(nwk_leaves), "source_file": self.nwk_path.name}
            self.species_names = nwk_leaves
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

    def _parse_newick_and_index_nodes(self):
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
                u.node_id = self.leaf_to_col[u.name]
            else:
                u.node_id = int_id
                if not u.name:
                    u.name = f"Internal_{int_id}"
                int_id += 1

        all_nodes = sorted(postorder, key=lambda node: node.node_id)
        node_rows, edge_rows = [], []
        for u in all_nodes:
            ntype = "leaf" if u.is_leaf() else ("root" if u.parent is None else "internal")
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
        return root, postorder, all_nodes, pd.DataFrame(node_rows), pd.DataFrame(edge_rows)

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

    def _build_covariance_and_cholesky(self):
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
        """Mô phỏng xuôi quá trình Chuyển động Brown (BM SDE) từ gốc xuống 148 nút lá."""
        if rng is None:
            rng = np.random.default_rng(42)
        node_vals = np.zeros((n_replicates, len(self.all_nodes)), dtype=np.float64)
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

    def simulate_dataset_cohort(
        self,
        n_samples: int = NUM_TEST_SAMPLES,
        seed: int = SEED_FIXED_TEST,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mô phỏng tập dữ liệu gồm `n_samples` mẫu độc lập theo đúng quy trình sinh dữ liệu
        của `calib_123.csv`:
          - mu_k ~ Uniform(-5.0, +5.0)
          - ln(sigma_k) ~ Uniform(ln(0.1), ln(5.0)) <=> sigma_k in [0.1, 5.0]
          - x_k = mu_k * 1_148 + sigma_k * L_S * z_k, với z_k ~ N(0, I_148).
        """
        rng = np.random.default_rng(seed)
        mu_vec = rng.uniform(PRIOR_MU_MIN, PRIOR_MU_MAX, size=n_samples)
        log_sig_vec = rng.uniform(
            math.log(PRIOR_SIGMA_MIN), math.log(PRIOR_SIGMA_MAX), size=n_samples
        )
        sig_vec = np.exp(log_sig_vec)
        z_mat = rng.normal(0.0, 1.0, size=(n_samples, self.n_leaves))
        x_mat = mu_vec[:, None] + sig_vec[:, None] * (z_mat @ self.L_S.T)
        return mu_vec, sig_vec, x_mat


def ensure_fixed_test_set_1000(
    builder: FromScratchPhylogenyBuilder,
    test_csv_path: Path = TEST_SIM_CSV_PATH,
) -> pd.DataFrame:
    """
    Đảm bảo tệp tập kiểm tra mô phỏng cố định `data/raw/test_sim_1000_fixed.csv`
    (1,000 mẫu, `SEED_FIXED_TEST = 2026`) luôn tồn tại và trả về DataFrame của tập Test.
    """
    if test_csv_path.exists():
        df_te = pd.read_csv(test_csv_path)
        if len(df_te) == NUM_TEST_SAMPLES:
            return df_te

    mu_te, sig_te, x_te = builder.simulate_dataset_cohort(
        n_samples=NUM_TEST_SAMPLES, seed=SEED_FIXED_TEST
    )
    cols = {"mu": mu_te, "sigma": sig_te}
    for j, sp_name in enumerate(builder.species_names):
        cols[sp_name] = x_te[:, j]
    df_te = pd.DataFrame(cols)
    df_te.to_csv(test_csv_path, index=False)
    return df_te


def run_step0_verification() -> Dict[str, float]:
    """Thực thi kiểm định Bước 0 và khởi tạo tập Test cố định 1,000 mẫu."""
    ensure_output_directories()
    builder = FromScratchPhylogenyBuilder()
    df_test = ensure_fixed_test_set_1000(builder)

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
    return {
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
        "n_fixed_test_samples": len(df_test),
    }


if __name__ == "__main__":
    info = run_step0_verification()
    print("===== KẾT QUẢ KIỂM ĐỊNH BƯỚC 0: ĐỒ THỊ CÂY PHẢ HỆ & TẬP TEST MÔ PHỎNG CỐ ĐỊNH =====")
    print(f"  * Số loài lá / nút / cạnh có hướng : {info['n_leaves']} lá | {info['n_total_nodes']} nút | {info['n_directed_edges']} cạnh")
    print(f"  * Hằng số cấu trúc cây             : ln det(S) = {info['logdet_S']:.6f}, vec11 = {info['vec11']:.6f}, v_rho = {info['v_root']:.6f}")
    print(f"  * Hệ số suy giảm phương sai mẫu    : kappa_T = {info['kappa_tree']:.6f}")
    print(f"  * Tập Test mô phỏng cố định        : {info['n_fixed_test_samples']:,d} mẫu (test_sim_1000_fixed.csv, SEED={SEED_FIXED_TEST})")
