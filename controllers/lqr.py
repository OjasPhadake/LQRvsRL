# controllers/lqr.py
#
# Linear Quadratic Regulator for the discrete-time double integrator.
#
# Theory recap (connecting to your optimal control background):
# ─────────────────────────────────────────────────────────────
# LQR minimises the infinite-horizon cost:
#
#   J = Σ_{k=0}^{∞} ( x_k' Q x_k  +  u_k' R u_k )
#
# For a linear system x_{k+1} = A x_k + B u_k, dynamic programming
# tells us the optimal cost-to-go (value function) has the form:
#
#   V*(x) = x' P x
#
# where P is the solution to the Discrete Algebraic Riccati Equation (DARE):
#
#   P = Q  +  A' P A  -  A' P B (R + B' P B)^{-1} B' P A
#
# Once P is known, the optimal control gain follows immediately:
#
#   K = (R + B' P B)^{-1} B' P A
#   u_k = -K x_k
#
# This is the Bellman optimality condition applied to a linear-quadratic
# problem. The DARE is just the fixed-point of the Bellman equation.
#
# Key connection to RL:
#   - P in LQR  ≡  the value function V*(s) in RL
#   - K in LQR  ≡  the optimal policy π*(s) in RL
#   - Both are derived from the same Bellman equation.
#   - LQR solves it analytically because dynamics (A, B) are known.
#   - PPO approximates it numerically because dynamics are unknown.

import numpy as np
from scipy.linalg import solve_discrete_are


class LQRController:
    """
    Solves the DARE and provides an optimal state-feedback controller.

    Usage
    -----
    ctrl = LQRController(A, B, Q, R)
    u    = ctrl(obs)          # callable: obs (2,) → action (1,)
    ctrl.print_summary()      # prints K, P, eigenvalues
    """

    def __init__(self, A, B, Q, R):
        """
        Parameters
        ----------
        A : (n, n) array   — state transition matrix
        B : (n, m) array   — control input matrix
        Q : (n, n) array   — state cost matrix  (must be positive semi-definite)
        R : (m, m) array   — control cost matrix (must be positive definite)
        """
        self.A = np.atleast_2d(A).astype(np.float64)
        self.B = np.atleast_2d(B).astype(np.float64)
        self.Q = np.atleast_2d(Q).astype(np.float64)
        self.R = np.atleast_2d(R).astype(np.float64)

        # ── Solve the DARE ─────────────────────────────────────────────────
        # scipy.linalg.solve_discrete_are solves:
        #   P = Q + A'PA - A'PB(R + B'PB)^{-1}B'PA
        # This is the discrete-time version of the continuous Riccati equation.
        # P is the unique positive-definite solution when (A,B) is stabilisable
        # and (A, Q^{1/2}) is detectable — both true for our double integrator.
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)

        # ── Compute optimal gain K ─────────────────────────────────────────
        # K = (R + B'PB)^{-1} B'PA
        # This is derived by taking ∂/∂u of the Bellman equation and setting
        # it to zero — exactly the same as solving the first-order optimality
        # condition (Pontryagin's minimum principle in discrete time).
        self.K = (
            np.linalg.inv(self.R + self.B.T @ self.P @ self.B)
            @ self.B.T @ self.P @ self.A
        )

        # ── Closed-loop system ─────────────────────────────────────────────
        # Under u = -Kx, the dynamics become:
        #   x_{k+1} = (A - BK) x_k
        # Stability requires all eigenvalues of (A - BK) inside unit circle.
        self.A_cl   = self.A - self.B @ self.K
        self.cl_eig = np.linalg.eigvals(self.A_cl)

        # ── Verify stability ───────────────────────────────────────────────
        if not np.all(np.abs(self.cl_eig) < 1.0):
            raise RuntimeError(
                f"LQR solution is not stabilising!\n"
                f"Closed-loop eigenvalues: {self.cl_eig}"
            )

    def __call__(self, obs):
        """
        Compute optimal control: u = -K x

        Conforms to the controller_fn(obs) → action signature used by
        DoubleIntegratorEnv.simulate(), so it can be passed in directly.

        Parameters
        ----------
        obs : array-like, shape (2,)   [position, velocity]

        Returns
        -------
        action : np.ndarray, shape (1,)
        """
        x = np.asarray(obs, dtype=np.float64)
        u = -(self.K @ x)           # shape (1,)
        return np.clip(u, -5.0, 5.0)

    def optimal_cost(self, x0):
        """
        Compute the theoretical infinite-horizon optimal cost from x0.

        For the infinite-horizon LQR, the optimal cost is:
            J*(x0) = x0' P x0
        This is the value function V*(x0) evaluated at the initial state.
        It is a lower bound — no controller can achieve less cost than this
        on this linear-quadratic problem.

        Parameters
        ----------
        x0 : array-like, shape (2,)

        Returns
        -------
        float : theoretical minimum cost J*
        """
        x0 = np.asarray(x0, dtype=np.float64)
        return float(x0 @ self.P @ x0)

    def print_summary(self):
        """Print a formatted summary of the LQR solution."""
        width = 52
        print()
        print("=" * width)
        print("  LQR SOLUTION SUMMARY")
        print("=" * width)

        print("\n  Cost matrices:")
        print(f"    Q = {self.Q.tolist()}")
        print(f"    R = {self.R.tolist()}")

        print("\n  DARE solution  P:")
        for row in self.P:
            print(f"    [{row[0]:+10.6f}  {row[1]:+10.6f}]")

        print("\n  Optimal gain  K:")
        print(f"    [{self.K[0,0]:+10.6f}  {self.K[0,1]:+10.6f}]")
        print(f"\n    → u = -{self.K[0,0]:.4f}·pos  –  {self.K[0,1]:.4f}·vel")

        print("\n  Open-loop eigenvalues  (A):")
        ol_eig = np.linalg.eigvals(self.A)
        for e in ol_eig:
            print(f"    {e:.6f}   |λ| = {abs(e):.6f}")

        print("\n  Closed-loop eigenvalues  (A − BK):")
        for e in self.cl_eig:
            print(f"    {e:.6f}   |λ| = {abs(e):.6f}")

        print(f"\n  Stable (all |λ| < 1): {np.all(np.abs(self.cl_eig) < 1.0)}")

        x0 = np.array([5.0, 0.0])
        print(f"\n  Theoretical optimal cost  J*(x0=[5,0]):")
        print(f"    x0' P x0 = {self.optimal_cost(x0):.4f}")
        print("=" * width)
        print()


def settling_time(positions, threshold=0.1, dt=0.1):
    """
    Find the first time k such that |position| < threshold for all k' >= k.

    Parameters
    ----------
    positions : array (T+1,)
    threshold : float, default 0.1 m
    dt        : float, timestep in seconds

    Returns
    -------
    float : settling time in seconds, or np.inf if never settled
    """
    for k in range(len(positions)):
        if np.all(np.abs(positions[k:]) < threshold):
            return k * dt
    return np.inf


def run_lqr():
    """
    Full LQR pipeline: solve → simulate → print metrics → save results.

    Called by run_lqr.py --lqr
    Also called by main_compare.py to get the LQR baseline.

    Returns
    -------
    dict : simulation results from env.simulate()
    LQRController : the fitted controller (K, P accessible as attributes)
    """
    import numpy as np
    from envs.double_integrator_env import DoubleIntegratorEnv
    from analysis.plots import plot_trajectory, C_LQR

    env = DoubleIntegratorEnv(max_steps=200)

    # ── Instantiate and solve ──────────────────────────────────────────────
    ctrl = LQRController(env.A, env.B, env.Q, env.R)
    ctrl.print_summary()

    # ── Simulate ───────────────────────────────────────────────────────────
    print("  Simulating LQR trajectory ...")
    results = env.simulate(ctrl, x0=[5.0, 0.0])

    # ── Metrics ───────────────────────────────────────────────────────────
    t_settle = settling_time(results["positions"])
    J_theory = ctrl.optimal_cost([5.0, 0.0])

    print("\n  SIMULATION METRICS")
    print("  " + "─" * 45)
    print(f"    Steps to termination : {results['n_steps']}")
    print(f"    Final position       : {results['positions'][-1]:+.6f} m")
    print(f"    Final velocity       : {results['velocities'][-1]:+.6f} m/s")
    print(f"    Settling time (±0.1) : {t_settle:.2f} s")
    print(f"    Total cost  J        : {results['total_cost']:.4f}")
    print(f"    Theoretical J*(x0'Px0): {J_theory:.4f}")
    print(f"    Suboptimality        : "
          f"{100*(results['total_cost']-J_theory)/J_theory:.2f}%")
    print(f"    Peak |u|             : {np.max(np.abs(results['actions'])):.4f}")
    print(f"    RMS  |u|             : "
          f"{np.sqrt(np.mean(results['actions']**2)):.4f}")
    print()

    # ── Save trajectory data ───────────────────────────────────────────────
    np.save("results/data/lqr_results.npy", results)
    print("  Trajectory saved → results/data/lqr_results.npy")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_trajectory(
        results,
        title     = r"LQR Controller — Double Integrator ($x_0=[5,0]$)",
        color     = C_LQR,
        save_path = "results/figures/lqr_trajectory.png",
    )

    return results, ctrl