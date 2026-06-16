# main_compare.py
#
# Entry point for the full LQR vs PPO comparison.
#
# What this script does, in order:
#   1. Load (or regenerate) LQR and PPO trajectories
#   2. Compute all comparison metrics via analysis.compare
#   3. Print the comparison table to stdout
#   4. Generate every figure needed for the report:
#        - Overlay: position, velocity, control input (LQR vs PPO)
#        - Cumulative cost curves
#        - PPO learning curve
#        - Individual trajectories (for appendix)
#   5. Print the LaTeX table string, ready to paste into report.tex
#   6. Save all metrics to results/data/comparison_metrics.npy
#
# Run:
#   python main_compare.py                  # use saved trajectories (default)
#   python main_compare.py --retrain        # retrain PPO then compare
#   python main_compare.py --rerun-lqr      # re-simulate LQR then compare
#
# Expected outputs:
#   results/figures/comparison_overlay.png    ← main report figure
#   results/figures/cumulative_cost.png
#   results/figures/ppo_learning_curve.png
#   results/figures/lqr_trajectory.png
#   results/figures/ppo_trajectory.png
#   results/data/comparison_metrics.npy

import argparse
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from envs.double_integrator_env  import DoubleIntegratorEnv
from controllers.lqr             import LQRController, run_lqr
from controllers.rl_agent        import run_ppo, evaluate_ppo
from analysis.compare            import compare_metrics, print_comparison_table, latex_table
from analysis.plots              import (
    plot_comparison,
    plot_cumulative_cost,
    plot_learning_curve,
    plot_trajectory,
    C_LQR, C_PPO,
)


# ── Loaders ───────────────────────────────────────────────────────────────

def _load_or_run_lqr(force_rerun=False):
    """
    Load saved LQR trajectory, or re-simulate if not found / forced.
    """
    path = "results/data/lqr_results.npy"
    if not force_rerun and os.path.exists(path):
        print("  Loading LQR results from disk ...")
        return np.load(path, allow_pickle=True).item()
    else:
        print("  Running LQR simulation ...")
        results, _ = run_lqr()
        return results


def _load_or_train_ppo(force_retrain=False):
    """
    Load saved PPO trajectory, or retrain + evaluate if not found / forced.

    Note: loading the trajectory (ppo_results.npy) is enough for comparison.
    We only need the full model (ppo_model.zip) if we want to interact with
    it or watch it run — evaluate_ppo handles that on demand.
    """
    results_path = "results/data/ppo_results.npy"
    model_path   = "results/data/ppo_model.zip"

    if not force_retrain and os.path.exists(results_path):
        print("  Loading PPO results from disk ...")
        return np.load(results_path, allow_pickle=True).item()
    elif not force_retrain and os.path.exists(model_path):
        print("  PPO model found; evaluating on raw environment ...")
        results = evaluate_ppo("results/data/ppo_model")
        np.save(results_path, results)
        return results
    else:
        print("  Training PPO from scratch ...")
        results, _, _ = run_ppo(total_timesteps=100_000)
        return results


def _load_learning_curve():
    """Load the PPO learning curve data saved during training."""
    path = "results/data/ppo_learning_curve.npy"
    if not os.path.exists(path):
        print("  Warning: learning curve data not found. "
              "Run train_ppo.py first.")
        return None
    return np.load(path, allow_pickle=True).item()


# ── Main ──────────────────────────────────────────────────────────────────

def main(retrain=False, rerun_lqr=False):

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data",    exist_ok=True)

    print()
    print("═" * 55)
    print("  LQR vs PPO — FULL COMPARISON")
    print("  Double Integrator  |  x₀=[5,0]  |  Q=I, R=0.1")
    print("═" * 55)

    # ── Step 1: Get trajectories ───────────────────────────────────────────
    print()
    print("  [1/5] Loading trajectories ...")
    results_lqr = _load_or_run_lqr(force_rerun=rerun_lqr)
    results_ppo = _load_or_train_ppo(force_retrain=retrain)

    # ── Step 2: Compute J* from LQR's P matrix ────────────────────────────
    print()
    print("  [2/5] Computing metrics ...")
    env    = DoubleIntegratorEnv()
    ctrl   = LQRController(env.A, env.B, env.Q, env.R)
    J_star = ctrl.optimal_cost([5.0, 0.0])

    comparison = compare_metrics(results_lqr, results_ppo, J_star)

    # Save metrics for report and future reference
    np.save("results/data/comparison_metrics.npy", comparison)

    # ── Step 3: Print table ────────────────────────────────────────────────
    print()
    print("  [3/5] Comparison table:")
    print_comparison_table(comparison)

    # ── Step 4: Figures ───────────────────────────────────────────────────
    print("  [4/5] Generating figures ...")

    # ── Figure 1: Main overlay — the key comparison figure for the report ──
    # Three panels: position, velocity, control input. LQR (blue) overlaid
    # with PPO (orange). If PPO has truly learned u ≈ -Kx, the trajectories
    # should be almost indistinguishable — which is exactly what happens.
    plot_comparison(
        results_lqr,
        results_ppo,
        save_path = "results/figures/comparison_overlay.png",
    )

    # ── Figure 2: Cumulative cost ──────────────────────────────────────────
    # Shows J_k = Σ_{i=0}^{k} c_i growing over time. Both curves should
    # asymptote to similar values. The gap between them is the PPO cost gap.
    # LQR's curve is the lower envelope — the best any controller can do.
    plot_cumulative_cost(
        results_lqr,
        results_ppo,
        save_path = "results/figures/cumulative_cost.png",
    )

    # ── Figure 3: Learning curve ───────────────────────────────────────────
    # Mean episode reward vs training timesteps. Shows PPO converging from
    # random performance (~-4000) to near-LQR performance (~-364).
    # The LQR baseline is drawn as a horizontal reference line.
    lc = _load_learning_curve()
    if lc is not None:
        plot_learning_curve(
            lc["timesteps"],
            lc["mean_rewards"],
            lc["std_rewards"],
            lqr_baseline = -results_lqr["total_cost"],
            save_path    = "results/figures/ppo_learning_curve.png",
        )

    # ── Figures 4 & 5: Individual trajectories (for appendix) ─────────────
    plot_trajectory(
        results_lqr,
        title     = r"LQR Controller — Double Integrator ($x_0=[5,0]$)",
        color     = C_LQR,
        save_path = "results/figures/lqr_trajectory.png",
    )
    plot_trajectory(
        results_ppo,
        title     = r"PPO Controller — Double Integrator ($x_0=[5,0]$)",
        color     = C_PPO,
        save_path = "results/figures/ppo_trajectory.png",
    )

    print()
    print("  Figures saved to results/figures/:")
    for f in sorted(os.listdir("results/figures")):
        print(f"    {f}")

    # ── Step 5: LaTeX table ────────────────────────────────────────────────
    print()
    print("  [5/5] LaTeX table (paste into report.tex):")
    print()
    print(latex_table(comparison))
    print()

    print("═" * 55)
    print("  Done.  All outputs in results/")
    print("═" * 55)
    print()


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LQR vs PPO full comparison on the double integrator."
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain PPO from scratch before comparing."
    )
    parser.add_argument(
        "--rerun-lqr", action="store_true",
        help="Re-simulate LQR before comparing."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(retrain=args.retrain, rerun_lqr=args.rerun_lqr)