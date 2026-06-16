# analysis/plots.py
#
# Publication-quality plotting utilities shared across the project.
#
# Design principle: every function takes a results dict (as returned by
# DoubleIntegratorEnv.simulate()) and an optional Axes/Figure argument.
# This makes it easy to compose multi-panel figures in main_compare.py
# without duplicating plotting logic.
#
# Colour palette — used consistently across all figures so the report
# is visually coherent:
#   LQR  → BLUE   (#2563eb)
#   PPO  → ORANGE (#ea580c)
#   Zero → GREY   (#6b7280)
#   Ref  → BLACK dashed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 10,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
    "figure.dpi"       : 150,
    "savefig.dpi"      : 200,
    "savefig.bbox"     : "tight",
})

# Colour constants — import these in other modules too for consistency
C_LQR  = "#2563eb"   # blue
C_PPO  = "#ea580c"   # orange
C_ZERO = "#6b7280"   # grey
C_REF  = "black"


# ── dt for time axis ───────────────────────────────────────────────────────
DT = 0.1  # seconds per step — must match DoubleIntegratorEnv


# ── Single-controller trajectory plot ─────────────────────────────────────

def plot_trajectory(results, title, color=C_LQR, save_path=None):
    """
    Three-panel plot: position | velocity | control input vs time.

    Used by run_lqr.py and can be called stand-alone for any controller.
    """
    n   = results["n_steps"]
    t   = np.arange(n + 1) * DT   # state time axis  (T+1 points)
    t_u = np.arange(n)     * DT   # action time axis (T points)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Position
    axes[0].plot(t, results["positions"], color=color, linewidth=2)
    axes[0].axhline(0, color=C_REF, linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Position  [m]")
    axes[0].set_xlim([0, t[-1]])

    # Velocity
    axes[1].plot(t, results["velocities"], color=color, linewidth=2)
    axes[1].axhline(0, color=C_REF, linewidth=0.8, linestyle="--", alpha=0.6)
    axes[1].set_ylabel("Velocity  [m/s]")
    axes[1].set_xlim([0, t[-1]])

    # Control input
    axes[2].step(t_u, results["actions"], color=color, linewidth=2, where="post")
    axes[2].axhline(0, color=C_REF, linewidth=0.8, linestyle="--", alpha=0.6)
    axes[2].set_ylabel("Control  u  [—]")
    axes[2].set_xlabel("Time  [s]")
    axes[2].set_xlim([0, t_u[-1]])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close(fig)


# ── Side-by-side comparison plot ──────────────────────────────────────────

def plot_comparison(results_lqr, results_ppo, save_path=None):
    """
    Three-panel overlay: LQR (blue) vs PPO (orange).

    This is the main figure for the report and main_compare.py.
    Each panel shows both controllers on the same axes for direct comparison.
    """
    n_lqr = results_lqr["n_steps"]
    n_ppo = results_ppo["n_steps"]

    t_lqr   = np.arange(n_lqr + 1) * DT
    t_ppo   = np.arange(n_ppo + 1) * DT
    t_u_lqr = np.arange(n_lqr)     * DT
    t_u_ppo = np.arange(n_ppo)     * DT

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    fig.suptitle(
        "LQR vs PPO — Double Integrator ($x_0 = [5,\\ 0]$)",
        fontsize=13, fontweight="bold"
    )

    # ── Position ──────────────────────────────────────────────────────────
    axes[0].plot(t_lqr, results_lqr["positions"],
                 color=C_LQR, linewidth=2.0, label="LQR")
    axes[0].plot(t_ppo, results_ppo["positions"],
                 color=C_PPO, linewidth=2.0, label="PPO", linestyle="--")
    axes[0].axhline(0, color=C_REF, linewidth=0.8, linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Position  [m]")
    axes[0].legend(loc="upper right")

    # ── Velocity ──────────────────────────────────────────────────────────
    axes[1].plot(t_lqr, results_lqr["velocities"],
                 color=C_LQR, linewidth=2.0, label="LQR")
    axes[1].plot(t_ppo, results_ppo["velocities"],
                 color=C_PPO, linewidth=2.0, label="PPO", linestyle="--")
    axes[1].axhline(0, color=C_REF, linewidth=0.8, linestyle=":", alpha=0.5)
    axes[1].set_ylabel("Velocity  [m/s]")
    axes[1].legend(loc="upper right")

    # ── Control input ─────────────────────────────────────────────────────
    axes[2].step(t_u_lqr, results_lqr["actions"],
                 color=C_LQR, linewidth=2.0, label="LQR", where="post")
    axes[2].step(t_u_ppo, results_ppo["actions"],
                 color=C_PPO, linewidth=2.0, label="PPO",
                 linestyle="--", where="post")
    axes[2].axhline(0, color=C_REF, linewidth=0.8, linestyle=":", alpha=0.5)
    axes[2].set_ylabel("Control  u  [—]")
    axes[2].set_xlabel("Time  [s]")
    axes[2].legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close(fig)


# ── Cumulative cost plot ───────────────────────────────────────────────────

def plot_cumulative_cost(results_lqr, results_ppo, save_path=None):
    """
    Cumulative cost  J_k = Σ_{i=0}^{k} c_i  vs time.

    The asymptote of each curve is the total cost J.
    LQR's curve should be a lower bound — any policy that knows the
    dynamics cannot do better (for a linear-quadratic problem).
    """
    cum_lqr = np.cumsum(results_lqr["stage_costs"])
    cum_ppo = np.cumsum(results_ppo["stage_costs"])

    t_lqr = np.arange(len(cum_lqr)) * DT
    t_ppo = np.arange(len(cum_ppo)) * DT

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_lqr, cum_lqr, color=C_LQR, linewidth=2, label="LQR")
    ax.plot(t_ppo, cum_ppo, color=C_PPO, linewidth=2,
            label="PPO", linestyle="--")

    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Cumulative Cost  $J_k$")
    ax.set_title("Cumulative Stage Cost vs Time")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close(fig)


# ── PPO learning curve ─────────────────────────────────────────────────────

def plot_learning_curve(timesteps, mean_rewards, std_rewards=None,
                        lqr_baseline=None, save_path=None):
    """
    Plot mean episode reward vs training timesteps during PPO training.

    timesteps    : list of int
    mean_rewards : list of float
    std_rewards  : list of float or None
    lqr_baseline : float or None
        If provided, draws a horizontal dashed line at the LQR episode
        reward. This gives a visual reference: PPO has "converged" when
        its learning curve reaches this line.
    """
    t = np.array(timesteps)
    m = np.array(mean_rewards)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, m, color=C_PPO, linewidth=2, label="PPO mean episode reward")

    if std_rewards is not None:
        s = np.array(std_rewards)
        ax.fill_between(t, m - s, m + s,
                        color=C_PPO, alpha=0.2, label="±1 std")

    if lqr_baseline is not None:
        ax.axhline(
            lqr_baseline,
            color     = C_LQR,
            linewidth = 1.5,
            linestyle = "--",
            label     = f"LQR reward ({lqr_baseline:.1f})",
        )

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("PPO Learning Curve — Double Integrator")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k"
    ))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close(fig)


# ── Explore dynamics (used by run_lqr.py for Phase 1 section) ─────────────

def plot_open_loop_exploration(results_dict_list, labels, colors,
                               title, save_path=None):
    """
    Overlay multiple trajectories on one 3-panel figure.

    Used when exploring the uncontrolled dynamics before introducing LQR.
    results_dict_list : list of results dicts from env.simulate()
    labels            : list of str
    colors            : list of colour strings
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for results, label, color in zip(results_dict_list, labels, colors):
        n   = results["n_steps"]
        t   = np.arange(n + 1) * DT
        t_u = np.arange(n)     * DT

        axes[0].plot(t,   results["positions"],  color=color, lw=2, label=label)
        axes[1].plot(t,   results["velocities"], color=color, lw=2, label=label)
        axes[2].step(t_u, results["actions"],    color=color, lw=2,
                     label=label, where="post")

    for ax in axes:
        ax.axhline(0, color=C_REF, lw=0.8, ls="--", alpha=0.5)

    axes[0].set_ylabel("Position  [m]")
    axes[1].set_ylabel("Velocity  [m/s]")
    axes[2].set_ylabel("Control  u  [—]")
    axes[2].set_xlabel("Time  [s]")

    for ax in axes[:2]:
        ax.legend(loc="upper right")
    axes[2].legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close(fig)