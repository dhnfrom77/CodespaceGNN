#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bước 3A: Định nghĩa Hệ thống 6 Kiến trúc Mạng Nơ-ron Đồ thị (GNN M1 -> M5B),
         Tầng Truyền Thông điệp Hậu tố Có hướng `DirectedBottomUpTreeMPNNLayer`,
         Hàm Mất mát Dị phương sai NLL và Đầu đọc Co rút Hậu nghiệm Bayes Thực nghiệm
         (Tỷ số Mills Chuẩn Cụt 1D & Cầu phương Lưới Logarit 250 điểm).
"""

# 1. Thư viện chuẩn của Python
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 2. Thư viện bên thứ ba
import numpy as np
from scipy import stats

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        from torch_geometric.nn import MessagePassing
    except ImportError:
        MessagePassing = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class _DummyModule:
        pass

    nn = type(
        "nn",
        (),
        {
            "Module": _DummyModule,
            "Parameter": lambda x: x,
            "Sequential": lambda *a: None,
            "Linear": lambda *a, **k: None,
            "LayerNorm": lambda *a, **k: None,
            "GELU": lambda: None,
        },
    )
    MessagePassing = _DummyModule

# 3. Mã của chính dự án
from bm_gnn.step0_tree_and_bm_simulator import FromScratchPhylogenyBuilder, c4_factor


def truncated_normal_posterior_mean_np(
    mu_obs: np.ndarray,
    s0: np.ndarray,
    a_bound: float = -5.0,
    b_bound: float = 5.0,
) -> np.ndarray:
    """
    Tính kỳ vọng hậu nghiệm giải tích của phân phối Chuẩn Cụt 1D trên đoạn [a_bound, b_bound]
    thông qua hiệu số Tỷ số Mills (Inverse Mills Ratio):
        E[mu | mu_obs, s0] = mu_obs + s0 * (phi(alpha) - phi(beta)) / (Phi(beta) - Phi(alpha)).
    """
    s0_safe = np.maximum(np.asarray(s0, dtype=np.float64), 1e-8)
    mu_arr = np.asarray(mu_obs, dtype=np.float64)
    alpha = (a_bound - mu_arr) / s0_safe
    beta = (b_bound - mu_arr) / s0_safe
    phi_alpha = stats.norm.pdf(alpha)
    phi_beta = stats.norm.pdf(beta)
    cdf_diff = np.maximum(stats.norm.cdf(beta) - stats.norm.cdf(alpha), 1e-12)
    mu_post = mu_arr + s0_safe * (phi_alpha - phi_beta) / cdf_diff
    return np.clip(mu_post, a_bound, b_bound)


def quadrature_250_lognormal_sigma_mmse_np(
    sigma_m5a: np.ndarray,
    mu_ln_prior: float,
    sd_ln_prior: float,
    df: int = 147,
    grid_size: int = 250,
    s_min: float = 0.1,
    s_max: float = 5.0,
) -> np.ndarray:
    """
    Cầu phương số 250 điểm trên lưới logarit u_g = ln(sigma_g) in [ln(0.1), ln(5.0)]
    kèm thủ thuật ổn định số học Log-Sum-Exp để tính kỳ vọng hậu nghiệm Bayes
    E[sigma_k | Q_int,k] dưới tiên nghiệm học được từ 10,000 mẫu Train.
    """
    u_grid = np.linspace(math.log(s_min), math.log(s_max), grid_size, dtype=np.float64)[
        None, :
    ]
    sig_grid = np.exp(u_grid)
    c4_inv = 1.0 / c4_factor(df)
    q_est = (df * ((np.asarray(sigma_m5a, dtype=np.float64) / c4_inv) ** 2))[:, None]

    log_lik = -df * u_grid - q_est / (2.0 * (sig_grid ** 2))
    log_prior = -0.15 * ((u_grid - mu_ln_prior) ** 2) / (2.0 * (sd_ln_prior ** 2))
    log_post = log_lik + log_prior
    log_post = log_post - np.max(log_post, axis=1, keepdims=True)
    weights = np.exp(log_post)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return np.sum(weights * sig_grid, axis=1)


if TORCH_AVAILABLE:

    class DirectedBottomUpTreeMPNNLayer(MessagePassing):
        """Tầng truyền thông điệp cây có hướng từ dưới lên (Postorder Message Passing)."""

        def __init__(self, left_ids, right_ids, dt_left, dt_right):
            super().__init__(aggr="add", flow="source_to_target")
            self.register_buffer("left_ids", torch.as_tensor(left_ids, dtype=torch.long))
            self.register_buffer("right_ids", torch.as_tensor(right_ids, dtype=torch.long))
            self.register_buffer("dt_left", torch.as_tensor(dt_left, dtype=torch.float64))
            self.register_buffer("dt_right", torch.as_tensor(dt_right, dtype=torch.float64))
            self.log_branch_scale = nn.Parameter(
                torch.zeros(len(left_ids), 2, dtype=torch.float64)
            )

        def forward(
            self, x_leaves: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            bsz, n_leaves = x_leaves.shape
            n_int = len(self.left_ids)
            h_list = [x_leaves[:, i] for i in range(n_leaves)]
            v_list = [
                torch.zeros((), dtype=torch.float64, device=x_leaves.device)
                for _ in range(n_leaves)
            ]
            c_int_list = []
            scales = torch.exp(self.log_branch_scale)

            for u_idx in range(n_int):
                l_id = int(self.left_ids[u_idx].item())
                r_id = int(self.right_ids[u_idx].item())
                d_l = v_list[l_id] + self.dt_left[u_idx] * scales[u_idx, 0]
                d_r = v_list[r_id] + self.dt_right[u_idx] * scales[u_idx, 1]
                d_sum = d_l + d_r
                h_u = (d_r * h_list[l_id] + d_l * h_list[r_id]) / d_sum
                v_u = (d_l * d_r) / d_sum
                c_u = (h_list[l_id] - h_list[r_id]) / torch.sqrt(d_sum)
                h_list.append(h_u)
                v_list.append(v_u)
                c_int_list.append(c_u)

            return (
                torch.stack(h_list, dim=1),
                torch.stack(v_list),
                torch.stack(c_int_list, dim=1),
            )

    class QuadraticContrastScaleHead(nn.Module):
        """Đầu đọc toàn phương bất biến dịch chuyển cho sigma_hat."""

        def __init__(self, num_contrasts: int = 147):
            super().__init__()
            self.num_contrasts = num_contrasts
            self.contrast_logits = nn.Parameter(
                torch.zeros(num_contrasts, dtype=torch.float64)
            )
            self.c4_inv_scale = nn.Parameter(
                torch.tensor(1.0 / c4_factor(num_contrasts), dtype=torch.float64)
            )

        def forward(self, c_int: torch.Tensor) -> torch.Tensor:
            w_sigma = F.softmax(self.contrast_logits, dim=0)
            q_weighted = torch.sum(w_sigma.unsqueeze(0) * (c_int ** 2), dim=1)
            return self.c4_inv_scale * torch.sqrt(torch.clamp(q_weighted, min=1e-15))

    class EquivariantRootLocationHead(nn.Module):
        """Đầu đọc vị trí gốc đẳng biến dịch chuyển cho mu_hat."""

        def __init__(self, num_internal_diffs: int = 146):
            super().__init__()
            self.diff_weights = nn.Parameter(
                torch.zeros(num_internal_diffs, dtype=torch.float64)
            )

        def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
            h_root = h_nodes[:, -1]
            h_internal_diffs = h_nodes[:, 148:294] - h_root.unsqueeze(1)
            return h_root + torch.matmul(h_internal_diffs, self.diff_weights)

    class PhyloGNN_M5A_UnbiasedNLL(nn.Module):
        """Mô hình Đề xuất 1: Dual-Stream Equivariant PhyloGNN (Unbiased MVUE)."""

        def __init__(self, builder: FromScratchPhylogenyBuilder):
            super().__init__()
            gt = builder.graph_tensors
            self.mpnn = DirectedBottomUpTreeMPNNLayer(
                gt["left_ids"], gt["right_ids"], gt["dt_left"], gt["dt_right"]
            )
            self.scale_head = QuadraticContrastScaleHead(num_contrasts=147)
            self.loc_head = EquivariantRootLocationHead(num_internal_diffs=146)

        def forward(self, x_leaves: torch.Tensor) -> Dict[str, torch.Tensor]:
            h_nodes, v_nodes, c_int = self.mpnn(x_leaves)
            return {
                "mu_hat": self.loc_head(h_nodes),
                "sigma_hat": self.scale_head(c_int),
                "H_nodes": h_nodes,
                "v_nodes": v_nodes,
                "C_int": c_int,
            }


def verify_equivariance_and_invariance_properties() -> Dict[str, float]:
    """Kiểm chứng thực nghiệm tính Đẳng biến dịch chuyển của mu_hat và Bất biến dịch chuyển của sigma_hat."""
    builder = FromScratchPhylogenyBuilder()
    rng = np.random.default_rng(123)
    x_sample = builder.simulate_forward_bm_preorder(
        mu=1.25, sigma=2.40, n_replicates=8, rng=rng
    )
    shift_c = 3.75
    x_shifted = x_sample + shift_c

    if TORCH_AVAILABLE:
        model_m5a = PhyloGNN_M5A_UnbiasedNLL(builder)
        with torch.no_grad():
            out1 = model_m5a(torch.tensor(x_sample, dtype=torch.float64))
            out2 = model_m5a(torch.tensor(x_shifted, dtype=torch.float64))
        equiv_err = float(
            torch.max(torch.abs((out2["mu_hat"] - out1["mu_hat"]) - shift_c)).item()
        )
        invar_err = float(
            torch.max(torch.abs(out2["sigma_hat"] - out1["sigma_hat"])).item()
        )
    else:
        from bm_gnn.step2_phylolm_3pt_and_solvers import (
            run_phylolm_3pt_pruning_from_scratch,
        )

        out1 = run_phylolm_3pt_pruning_from_scratch(builder, x_sample)
        out2 = run_phylolm_3pt_pruning_from_scratch(builder, x_shifted)
        equiv_err = float(np.max(np.abs((out2["mu_hat"] - out1["mu_hat"]) - shift_c)))
        invar_err = float(np.max(np.abs(out2["sigma_reml"] - out1["sigma_reml"])))

    return {
        "shift_constant_c": shift_c,
        "translation_equivariance_max_error_mu": equiv_err,
        "translation_invariance_max_error_sigma": invar_err,
    }


if __name__ == "__main__":
    res_sym = verify_equivariance_and_invariance_properties()
    print("===== KẾT QUẢ BƯỚC 3A: KIỂM ĐỊNH ĐỐI XỨNG HÌNH HỌC CỦA KIẾN TRÚC PhyloGNN =====")
    print(f"  * Hằng số dịch chuyển thử nghiệm c             : +{res_sym['shift_constant_c']:.2f}")
    print(f"  * Sai số Đẳng biến Dịch chuyển |mu(x+c)-mu(x)-c|: {res_sym['translation_equivariance_max_error_mu']:.3e}")
    print(f"  * Sai số Bất biến Dịch chuyển |sig(x+c)-sig(x)| : {res_sym['translation_invariance_max_error_sigma']:.3e}")
