#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bộ kiểm thử tự động (Unit Tests) cho dự án `CodebaseGNN`.
Chạy được bằng cả hai cách:
  - Cách 1 (pytest):   python -m pytest
  - Cách 2 (unittest): python -m unittest discover -s tests
"""

# 1. Thư viện chuẩn của Python
import math
import sys
import unittest
from pathlib import Path

# Đảm bảo thư mục `src/` nằm trong sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. Thư viện bên thứ ba
import numpy as np

# 3. Mã của chính dự án
from bm_gnn.step0_tree_and_bm_simulator import FromScratchPhylogenyBuilder, c4_factor
from bm_gnn.step2_phylolm_3pt_and_solvers import (
    run_cholesky_gls_and_pic_solvers,
    run_phylolm_3pt_pruning_from_scratch,
)
from bm_gnn.step3_gnn_architectures import (
    truncated_normal_posterior_mean_np,
    verify_equivariance_and_invariance_properties,
)


class TestBMGNNPipeline(unittest.TestCase):
    """Kiểm thử toàn diện cấu trúc đồ thị 295 nút, thuật toán phylolm và đối xứng GNN."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = FromScratchPhylogenyBuilder()

    def test_01_tree_combinatorics_and_ultrametricity(self) -> None:
        """Kiểm tra cây nhị phân chặt 148 loài có đúng 295 nút, 588 cạnh có hướng và T_i = 1.0."""
        self.assertEqual(self.builder.n_leaves, 148)
        self.assertEqual(len(self.builder.all_nodes), 295)
        self.assertEqual(len(self.builder.edges_df), 588)
        diag_s = np.diag(self.builder.S_mat)
        self.assertTrue(np.allclose(diag_s, 1.0, atol=1e-6))
        self.assertAlmostEqual(self.builder.vec11, 4.94635812, places=5)
        self.assertAlmostEqual(self.builder.v_root, 0.20216894, places=5)
        self.assertAlmostEqual(self.builder.kappa_tree, 0.615125, places=5)

    def test_02_phylolm_vs_cholesky_and_pic_machine_precision(self) -> None:
        """Kiểm chứng cắt tỉa 3 điểm phylolm khớp Cholesky GLS và PIC tới cấp độ máy 10^-13."""
        rng = np.random.default_rng(2026)
        x_sim = self.builder.simulate_forward_bm_preorder(
            mu=-1.8, sigma=2.5, n_replicates=16, rng=rng
        )
        res_phylo = run_phylolm_3pt_pruning_from_scratch(self.builder, x_sim)
        res_other = run_cholesky_gls_and_pic_solvers(self.builder, x_sim)

        self.assertLess(
            np.max(np.abs(res_phylo["mu_hat"] - res_other["mu_chol"])), 1e-13
        )
        self.assertLess(
            np.max(np.abs(res_phylo["mu_hat"] - res_other["mu_pic"])), 1e-13
        )
        self.assertLess(
            np.max(np.abs(res_phylo["Q_internal"] - res_other["Q_chol"])), 1e-11
        )
        self.assertLess(
            np.max(np.abs(res_phylo["Q_internal"] - res_other["Q_pic"])), 1e-11
        )

    def test_03_translation_equivariance_and_invariance(self) -> None:
        """Kiểm chứng mu_hat(x + c*1) = mu_hat(x) + c và sigma_hat(x + c*1) = sigma_hat(x)."""
        sym = verify_equivariance_and_invariance_properties()
        self.assertLess(sym["translation_equivariance_max_error_mu"], 1e-12)
        self.assertLess(sym["translation_invariance_max_error_sigma"], 1e-12)

    def test_04_truncated_normal_bayes_shrinkage(self) -> None:
        """Kiểm chứng đầu đọc Chuẩn Cụt [-5, 5] co rút ước lượng vọt biên về trong [-5, 5]."""
        mu_overshoot = np.array([-6.5, 0.0, +6.5])
        s0 = np.array([1.8, 0.5, 1.8])
        mu_shrunk = truncated_normal_posterior_mean_np(mu_overshoot, s0)
        self.assertGreater(mu_shrunk[0], -5.0)
        self.assertAlmostEqual(mu_shrunk[1], 0.0, places=7)
        self.assertLess(mu_shrunk[2], 5.0)
        self.assertAlmostEqual(c4_factor(147), 0.99830101, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
