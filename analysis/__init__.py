# analysis/__init__.py
from .plots import (
    plot_trajectory,
    plot_comparison,
    plot_cumulative_cost,
    plot_learning_curve,
    plot_open_loop_exploration,
    C_LQR, C_PPO, C_ZERO,
)
from .compare import (
    compute_metrics,
    compare_metrics,
    print_comparison_table,
    latex_table,
)