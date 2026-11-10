#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bộ kiểm thử tự động (Unit Tests) cho dự án `CodespaceGNN`.
Chạy bằng:
  - python -m pytest -v
  - hoặc: python -m unittest discover -s tests -v
"""

# 1. Thư viện chuẩn của Python
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. Thư viện bên thứ ba
import numpy as np

# 3. Mã của chính dự án
from bm_gnn.config_paths import FLAGSHIP_SCATTER_PATH, TEST_SIM_CSV_PATH, TRAIN_CALIB_CSV_PATH
from bm_gnn.step0_tree_and_bm_simulator import FromScratchPhylogenyBuilder, c4_factor
from bm_gnn.step1_data_diagnostics_and_star import load_train_and_fixed_test_datasets
from bm_gnn.step2_phylolm_3pt_and_solvers import (
    run_cholesky_gls_and_pic_solvers,
    run_phylolm_3pt_pruning_from_scratch,
)
from bm_gnn.step3_gnn_architectures import (
    truncated_normal_posterior_mean_np,
    verify_equivariance_and_invariance_properties,
)
from bm_gnn.step3_train_and_ablation import (
    plot_flagship_test1000_scatter,
    train_on_calib10000_and_evaluate_on_fixed_test1000,
)


class TestCodespaceGNNPipeline(unittest.TestCase):
    """Kiểm thử toàn diện cấu trúc đồ thị 295 nút, phylolm, GNN và giao thức Train=10,000 / Test=1,000."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = FromScratchPhylogenyBuilder()

    def test_01_tree_combinatorics_and_ultrametricity(self) -> None:
        """Kiểm tra cây nhị phân chặt 148 loài có đúng 295 nút, 588 cạnh có hướng và T_i = 1.0."""
        self.assertEqual(self.builder.n_leaves, 148)
        self.assertEqual(len(self.builder.all_nodes), 295)
        self.assertEqual(len(self.builder.edges_df), 588)
        self.assertTrue(np.allclose(np.diag(self.builder.S_mat), 1.0, atol=1e-6))
        self.assertAlmostEqual(self.builder.vec11, 4.94635812, places=5)
        self.assertAlmostEqual(self.builder.v_root, 0.20216894, places=5)

    def test_02_phylolm_vs_cholesky_and_pic_machine_precision(self) -> None:
        """Kiểm chứng cắt tỉa 3 điểm phylolm khớp Cholesky GLS và PIC tới cấp độ máy 10^-13."""
        rng = np.random.default_rng(2026)
        x_sim = self.builder.simulate_forward_bm_preorder(
            mu=-1.8, sigma=2.5, n_replicates=16, rng=rng
        )
        res_phylo = run_phylolm_3pt_pruning_from_scratch(self.builder, x_sim)
        res_other = run_cholesky_gls_and_pic_solvers(self.builder, x_sim)
        self.assertLess(np.max(np.abs(res_phylo["mu_hat"] - res_other["mu_chol"])), 1e-13)
        self.assertLess(np.max(np.abs(res_phylo["mu_hat"] - res_other["mu_pic"])), 1e-13)

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

    def test_05_ml_protocol_train10000_and_fixed_test1000(self) -> None:
        """
        Kiểm chứng giao thức Học máy:
          - Tập Train (`calib_123.csv`) có đúng 10,000 mẫu.
          - Tập Test cố định (`test_sim_1000_fixed.csv`) có đúng 1,000 mẫu độc lập hoàn toàn.
          - Trên tập Test cố định, GNN M5A khớp phylolm (r > 0.99999) và GNN M5B giảm mạnh MSE
            trên cả mu và sigma, đồng thời xuất thành công biểu đồ Scatter Plot chủ đạo.
        """
        df_tr, mu_tr, _, _, df_te, mu_te, _, _ = load_train_and_fixed_test_datasets(
            self.builder
        )
        self.assertEqual(len(df_tr), 10000)
        self.assertEqual(len(df_te), 1000)
        self.assertEqual(len(set(np.round(mu_tr, 8)).intersection(set(np.round(mu_te, 8)))), 0)

        eval_b = train_on_calib10000_and_evaluate_on_fixed_test1000(self.builder)
        fig_path = plot_flagship_test1000_scatter(eval_b)
        self.assertTrue(fig_path.exists())

        bench_df = eval_b["bench_df"]
        self.assertEqual(len(bench_df), 8)
        self.assertTrue((bench_df["Split"] == "Test_Sim_1000").all())
        pred_df = eval_b["pred_test_df"]
        self.assertEqual(len(pred_df), 1000)
        r_m5a_phylo = np.corrcoef(
            pred_df["mu_hat_phylolm"], pred_df["mu_hat_GNN_M5A"]
        )[0, 1]
        self.assertGreater(r_m5a_phylo, 0.99999)

        mse_mu_phy = np.mean((pred_df["mu_hat_phylolm"] - pred_df["mu_true"]) ** 2)
        mse_mu_m5b = np.mean((pred_df["mu_hat_GNN_M5B"] - pred_df["mu_true"]) ** 2)
        self.assertLess(mse_mu_m5b, 0.75 * mse_mu_phy)

        mse_sig_phy = np.mean((pred_df["sigma_hat_phylolm"] - pred_df["sigma_true"]) ** 2)
        mse_sig_m5b = np.mean((pred_df["sigma_hat_GNN_M5B"] - pred_df["sigma_true"]) ** 2)
        self.assertLess(mse_sig_m5b, mse_sig_phy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
