# run_lqr.py
#
# Entry point for the LQR side of the project.
#
# Current state (Phase 1): explores the open-loop dynamics.
# After Phase 2: imports controllers.lqr and runs the LQR controller.
#
# Run:
#   python run_lqr.py              # Phase 1: dynamics exploration
#   python run_lqr.py --lqr        # Phase 2: LQR solve + simulation (TODO)

import argparse
import numpy as np

from envs.double_integrator_env import DoubleIntegratorEnv
from analysis.plots import (
    plot_open_loop_exploration,
    plot_trajectory,
    C_LQR, C_ZERO,
)

# ── Controller definitions ─────────────────────────────────────────────────
# These are deliberately simple — their purpose is to reveal the system's
# natural dynamics before any optimal controller is applied.


def zero_controller(obs):
    """
    u = 0 always.

    With no input, x_{k+1} = A x_k.
    Since A = [[1, 0.1], [0, 1]], both eigenvalues are +1 (unit circle).
    The system is marginally stable:
      - velocity stays at 0 (no force to change it)
      - position stays at its initial value (no velocity to move it)
    Starting from [5, 0] → position stays at 5.0 forever.
    This reveals that the system needs an active controller to reach [0,0].
    """
    return np.array([0.0])


def bang_bang_position_only(obs):
    """
    u = -sign(position) * 2.0

    Push opposite to position. Ignores velocity entirely.
    This will overshoot: position crosses zero, sign flips, pushes back.
    Result: sustained oscillation that never settles.

    This is the 1D analogue of the naive CartPole controller from Project 1
    that only used theta without theta_dot. The lesson is the same:
    you must feed back the full state, not just position.
    """
    position = float(obs[0])
    return np.array([-np.sign(position) * 2.0])


def pd_controller(obs):
    """
    u = -k_p * position - k_d * velocity

    A hand-tuned proportional-derivative controller.
    Using the full state [position, velocity] should damp the oscillations
    seen in bang_bang_position_only.

    k_p = 1.5, k_d = 2.0 are chosen by trial and error.
    LQR will analytically compute the optimal gains — compare K_lqr
    against [k_p, k_d] in Phase 2 to see how close hand-tuning gets.
    """
    position = float(obs[0])
    velocity = float(obs[1])
    k_p, k_d = 1.5, 2.0
    u = -k_p * position - k_d * velocity
    return np.array([np.clip(u, -5.0, 5.0)])


# ── Dynamics exploration ───────────────────────────────────────────────────

def explore_dynamics():
    """
    Run the system under three simple controllers and overlay trajectories.

    Goal: build intuition for the double integrator before solving LQR.
    Observations to make while reading the plots:
      1. Zero input     → position constant at 5.0. Neutral stability.
      2. Sign(position) → oscillates, never settles. Velocity ignored.
      3. PD controller  → settles, but sub-optimally. Full state used.
    """
    env = DoubleIntegratorEnv(max_steps=200)

    print("\n" + "=" * 55)
    print("  DYNAMICS EXPLORATION — Double Integrator")
    print("=" * 55)
    print(f"  System: x_{{k+1}} = A x_k + B u_k")
    print(f"  A = [[1, 0.1], [0, 1]]    B = [[0], [0.1]]")
    print(f"  x0 = [5.0, 0.0]           dt = 0.1 s")
    print(f"  Eigenvalues of A: {np.linalg.eigvals(env.A)}")
    print()

    controllers = [
        (zero_controller,          "Zero input (u=0)",              C_ZERO),
        (bang_bang_position_only,  "Sign controller (pos only)",    "#dc2626"),
        (pd_controller,            "PD controller (full state)",     C_LQR),
    ]

    results_list = []
    labels       = []
    colors       = []

    for ctrl_fn, label, color in controllers:
        results = env.simulate(ctrl_fn)
        results_list.append(results)
        labels.append(label)
        colors.append(color)
        _print_summary(label, results)

    # Overlay plot — all three controllers on the same axes
    plot_open_loop_exploration(
        results_list, labels, colors,
        title = "Open-Loop Dynamics Exploration — Double Integrator",
        save_path = "results/figures/dynamics_exploration.png",
    )

    # Individual plot for PD controller — cleaner for the report
    plot_trajectory(
        results_list[2],
        title     = "PD Controller — Hand-Tuned ($k_p=1.5,\\ k_d=2.0$)",
        color     = C_LQR,
        save_path = "results/figures/pd_controller.png",
    )

    print("\n  PHASE 1 TAKEAWAYS")
    print("  " + "─" * 50)
    print("  1. Zero input: position stays at 5.0 (neutral stability).")
    print("     Eigenvalues of A are both +1 — on the unit circle.")
    print("     Not asymptotically stable. Needs active control.")
    print()
    print("  2. Sign controller: oscillates indefinitely.")
    print("     Ignoring velocity = no damping. Same failure mode as")
    print("     the naive CartPole theta-only controller in Project 1.")
    print()
    print("  3. PD controller: settles, but what are the optimal gains?")
    print("     LQR (Phase 2) answers this analytically by solving the")
    print("     discrete-time algebraic Riccati equation (DARE).")
    print()
    print("  Next: run_lqr.py --lqr   (after building controllers/lqr.py)")
    print()


# ── Summary printer ────────────────────────────────────────────────────────

def _print_summary(label, results):
    print(f"  {'─'*48}")
    print(f"  {label}")
    print(f"  {'─'*48}")
    print(f"    Steps run          : {results['n_steps']}")
    print(f"    Final position     : {results['positions'][-1]:+.4f} m")
    print(f"    Final velocity     : {results['velocities'][-1]:+.4f} m/s")
    print(f"    Total cost  J      : {results['total_cost']:.4f}")
    print(f"    Peak |u|           : {np.max(np.abs(results['actions'])):.4f}")
    print()


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LQR vs RL — run_lqr.py"
    )
    parser.add_argument(
        "--lqr", action="store_true",
        help="Run LQR controller (requires controllers/lqr.py — Phase 2)"
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.lqr:
        # Phase 2 — solve DARE, compute K, simulate u = -Kx
        from controllers.lqr import run_lqr
        run_lqr()
    else:
        # Phase 1 — dynamics exploration (no optimal controller yet)
        explore_dynamics()