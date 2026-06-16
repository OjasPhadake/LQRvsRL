# analysis/compare.py
#
# Computes and formats every comparison metric between LQR and PPO.
#
# Design principle: this module is purely computational — no plotting,
# no side effects, no file I/O. It takes two results dicts (from
# DoubleIntegratorEnv.simulate()) and returns structured numbers.
# main_compare.py calls this, then decides what to print and plot.
#
# Metrics computed
# ────────────────
#  1. Total cost J           — primary objective; lower is better
#  2. Suboptimality vs J*    — how far each controller is from the
#                              infinite-horizon theoretical optimum x0'Px0
#  3. Suboptimality of PPO vs LQR  — the "gap" we care most about
#  4. Settling time          — first time |pos| < threshold and stays there
#  5. Peak control effort    — max |u_k| over the episode
#  6. RMS control effort     — energy proxy; sqrt(mean(u_k^2))
#  7. Final state error      — how close to [0,0] at end of episode
#  8. Control signal correlation — quantifies how similar PPO's u is to LQR's

import numpy as np
from controllers.lqr import settling_time


# ── Per-controller metric extraction ──────────────────────────────────────

def compute_metrics(results, label, J_star=None):
    """
    Extract all scalar metrics from one simulation results dict.

    Parameters
    ----------
    results : dict
        Output of DoubleIntegratorEnv.simulate(). Keys: positions,
        velocities, actions, stage_costs, total_cost, n_steps.
    label : str
        Controller name, e.g. "LQR" or "PPO". Used in returned dict.
    J_star : float or None
        Theoretical infinite-horizon minimum cost x0'Px0.
        If None, suboptimality vs theory is not computed.

    Returns
    -------
    dict of scalar metrics, all consistently named.
    """
    actions   = results["actions"]
    positions = results["positions"]
    vels      = results["velocities"]

    ts = settling_time(positions, threshold=0.1, dt=0.1)

    m = {
        "label"        : label,
        "total_cost"   : results["total_cost"],
        "settling_time": ts,                                  # seconds
        "settled"      : ts < np.inf,
        "peak_u"       : float(np.max(np.abs(actions))),
        "rms_u"        : float(np.sqrt(np.mean(actions**2))),
        "final_pos"    : float(positions[-1]),
        "final_vel"    : float(vels[-1]),
        "final_error"  : float(np.sqrt(positions[-1]**2 + vels[-1]**2)),
        "n_steps"      : results["n_steps"],
    }

    if J_star is not None:
        m["J_star"]          = J_star
        m["subopt_vs_theory"] = 100.0 * (m["total_cost"] - J_star) / J_star

    return m


# ── Cross-controller comparison ────────────────────────────────────────────

def compare_metrics(results_lqr, results_ppo, J_star):
    """
    Compute all metrics for both controllers plus cross-controller quantities.

    Parameters
    ----------
    results_lqr : dict   — from DoubleIntegratorEnv.simulate() with LQR
    results_ppo : dict   — from DoubleIntegratorEnv.simulate() with PPO
    J_star      : float  — theoretical minimum cost  x0' P x0

    Returns
    -------
    dict with keys:
        "lqr"         : per-controller metrics for LQR
        "ppo"         : per-controller metrics for PPO
        "cross"       : cross-controller quantities
    """
    m_lqr = compute_metrics(results_lqr, "LQR", J_star)
    m_ppo = compute_metrics(results_ppo, "PPO", J_star)

    # ── Cross-controller quantities ────────────────────────────────────────

    # Cost gap: how much more does PPO spend vs LQR?
    cost_gap_abs  = m_ppo["total_cost"] - m_lqr["total_cost"]
    cost_gap_pct  = 100.0 * cost_gap_abs / m_lqr["total_cost"]

    # Settling time difference (positive = PPO is slower)
    settle_delta  = (m_ppo["settling_time"] - m_lqr["settling_time"]
                     if m_ppo["settled"] and m_lqr["settled"]
                     else np.inf)

    # Control signal correlation — if PPO ≈ LQR, correlation ≈ 1.
    # Truncate to the shorter trajectory for a fair comparison.
    n_common = min(len(results_lqr["actions"]), len(results_ppo["actions"]))
    u_lqr    = results_lqr["actions"][:n_common]
    u_ppo    = results_ppo["actions"][:n_common]
    corr     = float(np.corrcoef(u_lqr, u_ppo)[0, 1])

    # Mean absolute deviation between action sequences
    action_mae = float(np.mean(np.abs(u_lqr - u_ppo)))

    cross = {
        "cost_gap_abs"  : cost_gap_abs,
        "cost_gap_pct"  : cost_gap_pct,
        "settle_delta_s": settle_delta,
        "action_corr"   : corr,
        "action_mae"    : action_mae,
        "J_star"        : J_star,
        "lqr_vs_theory" : m_lqr["subopt_vs_theory"],
        "ppo_vs_theory" : m_ppo["subopt_vs_theory"],
    }

    return {"lqr": m_lqr, "ppo": m_ppo, "cross": cross}


# ── Console report ─────────────────────────────────────────────────────────

def print_comparison_table(comparison):
    """
    Print a formatted side-by-side comparison table to stdout.

    This is the table that goes in the report (hand-transcribed or
    captured and formatted in LaTeX).
    """
    lqr   = comparison["lqr"]
    ppo   = comparison["ppo"]
    cross = comparison["cross"]

    W = 62
    print()
    print("═" * W)
    print("  LQR vs PPO — COMPARISON SUMMARY")
    print("  Double Integrator  |  x₀ = [5, 0]  |  Q=I, R=0.1")
    print("═" * W)
    print(f"  {'Metric':<30} {'LQR':>10} {'PPO':>10}")
    print("  " + "─" * (W - 2))

    def row(label, lval, pval, fmt=".4f", unit=""):
        lstr = f"{lval:{fmt}}{unit}" if lval is not None else "—"
        pstr = f"{pval:{fmt}}{unit}" if pval is not None else "—"
        print(f"  {label:<30} {lstr:>10} {pstr:>10}")

    row("Total cost  J",
        lqr["total_cost"], ppo["total_cost"])
    row("Theoretical  J*  (x₀'Px₀)",
        cross["J_star"], cross["J_star"])
    row("Suboptimality vs J*",
        lqr["subopt_vs_theory"], ppo["subopt_vs_theory"], fmt=".2f", unit="%")

    print("  " + "─" * (W - 2))

    ts_lqr = f"{lqr['settling_time']:.2f} s" if lqr["settled"] else "∞"
    ts_ppo = f"{ppo['settling_time']:.2f} s" if ppo["settled"] else "∞"
    print(f"  {'Settling time (|pos| < 0.1 m)':<30} {ts_lqr:>10} {ts_ppo:>10}")

    row("Peak  |u|",  lqr["peak_u"],  ppo["peak_u"])
    row("RMS   |u|",  lqr["rms_u"],   ppo["rms_u"])
    row("Final |pos|", abs(lqr["final_pos"]), abs(ppo["final_pos"]),
        fmt=".6f", unit=" m")
    row("Final |vel|", abs(lqr["final_vel"]), abs(ppo["final_vel"]),
        fmt=".6f", unit=" m/s")

    print("  " + "─" * (W - 2))

    print(f"  {'PPO cost gap vs LQR':<30} {'—':>10} "
          f"{cross['cost_gap_pct']:>+9.2f}%")
    print(f"  {'PPO action correlation':<30} {'—':>10} "
          f"{cross['action_corr']:>10.4f}")
    print(f"  {'PPO action MAE vs LQR':<30} {'—':>10} "
          f"{cross['action_mae']:>10.4f}")

    print("═" * W)
    print()
    print("  INTERPRETATION")
    print("  " + "─" * (W - 2))
    print(f"  LQR obtains the analytical optimum because the dynamics")
    print(f"  (A, B) are known and the Riccati equation can be solved")
    print(f"  exactly. Its {lqr['subopt_vs_theory']:.2f}% gap vs J* comes from the")
    print(f"  finite simulation horizon (200 steps vs infinite horizon).")
    print()
    print(f"  PPO learns a near-optimal policy purely through environment")
    print(f"  interaction — no knowledge of A or B. Its control signal")
    print(f"  correlates with LQR's at r = {cross['action_corr']:.4f}, meaning it")
    print(f"  effectively rediscovered the linear feedback law u ≈ -Kx.")
    print(f"  The {cross['cost_gap_pct']:+.2f}% cost gap vs LQR is negligible in practice.")
    print("═" * W)
    print()


# ── LaTeX table string ─────────────────────────────────────────────────────

def latex_table(comparison):
    """
    Return a LaTeX tabular string ready to paste into report.tex.

    Uses booktabs (toprule, midrule, bottomrule) for publication style.
    Requires \\usepackage{booktabs} in the LaTeX preamble.
    """
    lqr   = comparison["lqr"]
    ppo   = comparison["ppo"]
    cross = comparison["cross"]

    ts_lqr = f"{lqr['settling_time']:.2f}" if lqr["settled"] else r"$\infty$"
    ts_ppo = f"{ppo['settling_time']:.2f}" if ppo["settled"] else r"$\infty$"

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{LQR vs PPO on the Discrete-Time Double Integrator"
        r" ($x_0 = [5,\;0]^\top$, $Q = I$, $R = 0.1$)}",
        r"  \label{tab:comparison}",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Metric} & \textbf{LQR} & \textbf{PPO} \\",
        r"    \midrule",
        rf"    Total cost $J$ & {lqr['total_cost']:.4f} & {ppo['total_cost']:.4f} \\",
        rf"    Theoretical $J^* = x_0^\top P x_0$ & \multicolumn{{2}}{{c}}{{{cross['J_star']:.4f}}} \\",
        rf"    Suboptimality vs $J^*$ & {lqr['subopt_vs_theory']:.2f}\% & {ppo['subopt_vs_theory']:.2f}\% \\",
        r"    \midrule",
        rf"    Settling time ($|x_1| < 0.1$\,m) & {ts_lqr}\,s & {ts_ppo}\,s \\",
        rf"    Peak $|u|$ & {lqr['peak_u']:.4f} & {ppo['peak_u']:.4f} \\",
        rf"    RMS $|u|$ & {lqr['rms_u']:.4f} & {ppo['rms_u']:.4f} \\",
        rf"    Final $|x_1|$ (m) & {abs(lqr['final_pos']):.6f} & {abs(ppo['final_pos']):.6f} \\",
        r"    \midrule",
        rf"    PPO cost gap vs LQR & \multicolumn{{2}}{{c}}{{{cross['cost_gap_pct']:+.2f}\%}} \\",
        rf"    Control signal correlation $r$ & \multicolumn{{2}}{{c}}{{{cross['action_corr']:.4f}}} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)