"""
Gói mã lõi `bm_gnn` trong `CodespaceGNN` (Train = 10,000 mẫu `calib_123.csv`,
Test = 1,000 mẫu mô phỏng cố định `test_sim_1000_fixed.csv`).
"""

from bm_gnn.config_paths import (
    ARCH_DIAGRAMS_DIR,
    CALIB_CSV_PATH,
    FIGURES_DIR,
    FLAGSHIP_SCATTER_PATH,
    META_PATH,
    NUM_TEST_SAMPLES,
    NUM_TRAIN_SAMPLES,
    NWK_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SEED_FIXED_TEST,
    TEST_SIM_CSV_PATH,
    TRAIN_CALIB_CSV_PATH,
)
from bm_gnn.step0_tree_and_bm_simulator import (
    FromScratchPhylogenyBuilder,
    PhyloTreeNode,
    c4_factor,
    compute_reusable_metrics,
    ensure_fixed_test_set_1000,
    run_step0_verification,
)
from bm_gnn.step1_data_diagnostics_and_star import (
    load_and_audit_calib123,
    load_train_and_fixed_test_datasets,
    run_star_baseline,
    run_step1_pipeline,
)
from bm_gnn.step2_phylolm_3pt_and_solvers import (
    run_cholesky_gls_and_pic_solvers,
    run_phylolm_3pt_pruning_from_scratch,
    run_step2_pipeline,
)
from bm_gnn.step3_gnn_architectures import (
    TORCH_AVAILABLE,
    quadrature_250_lognormal_sigma_mmse_np,
    truncated_normal_posterior_mean_np,
    verify_equivariance_and_invariance_properties,
)
from bm_gnn.step3_train_and_ablation import (
    load_or_verify_step3_ablation_table,
    plot_flagship_test1000_scatter,
    run_step3_pipeline,
    train_on_calib10000_and_evaluate_on_fixed_test1000,
)
from bm_gnn.step4_multiscale_and_synthesis import (
    load_multiscale_benchmark_table,
    run_multiscale_evaluation_and_regenerate_figures,
    run_step4_pipeline,
)

__all__ = [
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "FIGURES_DIR",
    "ARCH_DIAGRAMS_DIR",
    "NWK_PATH",
    "META_PATH",
    "TRAIN_CALIB_CSV_PATH",
    "CALIB_CSV_PATH",
    "TEST_SIM_CSV_PATH",
    "FLAGSHIP_SCATTER_PATH",
    "SEED_FIXED_TEST",
    "NUM_TRAIN_SAMPLES",
    "NUM_TEST_SAMPLES",
    "PhyloTreeNode",
    "FromScratchPhylogenyBuilder",
    "c4_factor",
    "compute_reusable_metrics",
    "ensure_fixed_test_set_1000",
    "run_step0_verification",
    "load_and_audit_calib123",
    "load_train_and_fixed_test_datasets",
    "run_star_baseline",
    "run_step1_pipeline",
    "run_phylolm_3pt_pruning_from_scratch",
    "run_cholesky_gls_and_pic_solvers",
    "run_step2_pipeline",
    "TORCH_AVAILABLE",
    "truncated_normal_posterior_mean_np",
    "quadrature_250_lognormal_sigma_mmse_np",
    "verify_equivariance_and_invariance_properties",
    "train_on_calib10000_and_evaluate_on_fixed_test1000",
    "plot_flagship_test1000_scatter",
    "run_step3_pipeline",
    "run_multiscale_evaluation_and_regenerate_figures",
    "run_step4_pipeline",
]
