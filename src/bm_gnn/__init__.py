"""
Gói mã lõi `bm_gnn` cho đề tài:
Ước Lượng Đồng Thời Hai Tham Số (mu, sigma) Của Mô Hình Chuyển Động Brown
Trên Cây Phả Hệ Họ Kiến (Formicidae, n = 148 Loài) Bằng Thuật Toán `phylolm`
Và Mạng Nơ-Ron Đồ Thị Đẳng Biến Hai Luồng (`PhyloGNN`).
"""

from bm_gnn.config_paths import (
    ARCH_DIAGRAMS_DIR,
    CALIB_CSV_PATH,
    FIGURES_DIR,
    META_PATH,
    NWK_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from bm_gnn.step0_tree_and_bm_simulator import (
    FromScratchPhylogenyBuilder,
    PhyloTreeNode,
    c4_factor,
    compute_reusable_metrics,
    run_step0_verification,
)
from bm_gnn.step1_data_diagnostics_and_star import (
    load_and_audit_calib123,
    plot_figure0bc_horizontal,
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
    run_step3_pipeline,
)
from bm_gnn.step4_multiscale_and_synthesis import (
    load_multiscale_benchmark_table,
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
    "CALIB_CSV_PATH",
    "PhyloTreeNode",
    "FromScratchPhylogenyBuilder",
    "c4_factor",
    "compute_reusable_metrics",
    "run_step0_verification",
    "load_and_audit_calib123",
    "run_star_baseline",
    "plot_figure0bc_horizontal",
    "run_step1_pipeline",
    "run_phylolm_3pt_pruning_from_scratch",
    "run_cholesky_gls_and_pic_solvers",
    "run_step2_pipeline",
    "TORCH_AVAILABLE",
    "truncated_normal_posterior_mean_np",
    "quadrature_250_lognormal_sigma_mmse_np",
    "verify_equivariance_and_invariance_properties",
    "load_or_verify_step3_ablation_table",
    "run_step3_pipeline",
    "load_multiscale_benchmark_table",
    "run_step4_pipeline",
]
