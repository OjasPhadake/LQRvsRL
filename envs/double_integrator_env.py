# envs/double_integrator_env.py
#
# The shared simulator for the entire project.
# Both LQR (controllers/lqr.py) and PPO (controllers/rl_agent.py) run
# through this exact environment, which is what makes the comparison fair.
#
# System: discrete-time double integrator
#
#   x_{k+1} = A x_k + B u_k
#
#   A = [[1, 0.1],    B = [[0.0],
#        [0, 1.0]]         [0.1]]
#
# Physical interpretation (dt = 0.1 s):
#   x[0] = position  (m)
#   x[1] = velocity  (m/s)
#   u    = force/acceleration input  (continuous scalar)
#
#   position_{k+1} = position_k + 0.1 * velocity_k
#   velocity_{k+1} = velocity_k + 0.1 * u_k
#
# This is the simplest second-order system that is:
#   - marginally stable (eigenvalues of A are both +1, on the unit circle)
#   - fully controllable (rank([B, AB]) = 2)
#   - a standard benchmark in both control theory and RL
#
# Goal: drive x → [0, 0] from x0 = [5.0, 0.0]

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DoubleIntegratorEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, max_steps=200, x_limit=20.0):
        """
        Parameters
        ----------
        max_steps : int
            Episode length. 200 steps × dt=0.1 s = 20 seconds of simulation.
            LQR typically settles in ~3-5 s; PPO should learn to do the same.
        x_limit : float
            If |position| exceeds this, the episode terminates early.
            Acts as a safety boundary — both controllers must keep the
            system within bounds.
        """
        super().__init__()

        # ── Dynamics matrices ──────────────────────────────────────────────
        self.A = np.array([[1.0, 0.1],
                           [0.0, 1.0]], dtype=np.float64)

        self.B = np.array([[0.0],
                           [0.1]], dtype=np.float64)

        # ── Cost matrices ──────────────────────────────────────────────────
        # Q and R define the LQR objective:
        #   J = Σ_{k=0}^{∞} ( x_k' Q x_k  +  u_k' R u_k )
        #
        # These same matrices define the RL reward:
        #   r_k = -(x_k' Q x_k  +  u_k' R u_k)
        #
        # Using identical Q, R in both controllers is the conceptual core
        # of this project: same cost function, two different solvers.
        #
        # Q = I  →  penalise position² and velocity² equally
        # R = 0.1 →  small control penalty, allows the controller to
        #             use larger forces to settle quickly
        self.Q = np.eye(2, dtype=np.float64)
        self.R = np.array([[0.1]], dtype=np.float64)

        # ── Episode parameters ─────────────────────────────────────────────
        self.max_steps = max_steps
        self.x_limit   = x_limit

        # ── Gymnasium spaces ───────────────────────────────────────────────
        # Observation: [position, velocity]
        obs_high = np.array([x_limit, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low  = -obs_high,
            high =  obs_high,
            dtype = np.float32
        )

        # Action: continuous scalar force in [-5, 5]
        # Continuous because LQR produces a continuous u = -Kx.
        # Matching action spaces means we can overlay control signals
        # on the same plot without any discretisation artefacts.
        self.action_space = spaces.Box(
            low  = np.array([-5.0], dtype=np.float32),
            high = np.array([ 5.0], dtype=np.float32),
            dtype = np.float32
        )

        # Internal state — set properly in reset()
        self.state    = None
        self.step_num = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Fixed initial condition for every episode.
        # x0 = [5.0, 0.0]: system starts 5 m from origin with zero velocity.
        # Keeping x0 fixed means every training episode starts at the same
        # point, which is intentional — we want the agent to learn to
        # stabilise this specific operating condition.
        self.state    = np.array([5.0, 0.0], dtype=np.float64)
        self.step_num = 0

        return self.state.astype(np.float32), {}

    def step(self, action):
        # ── Parse and clip action ──────────────────────────────────────────
        # PPO outputs a 1-D array from a Gaussian policy.
        # LQR computes u = -K @ x, passed in as a 1-D array.
        # np.atleast_1d handles both scalar and array inputs.
        u = np.clip(
            np.atleast_1d(action).astype(np.float64),
            -5.0, 5.0
        )

        x = self.state  # shape (2,)

        # ── Apply dynamics ─────────────────────────────────────────────────
        # x_{k+1} = A x_k + B u_k
        # B @ u : (2,1) @ (1,) → (2,)
        x_next = self.A @ x + (self.B @ u).flatten()

        # ── Stage cost and reward ──────────────────────────────────────────
        # LQR stage cost: c_k = x' Q x + u' R u
        # RL  reward:     r_k = -c_k
        #
        # Maximising Σ r_k  ≡  minimising Σ c_k
        # This equivalence is the deepest link between RL and optimal control.
        stage_cost = float(x @ self.Q @ x  +  u @ self.R @ u)
        reward     = -stage_cost

        # ── Update internal state ──────────────────────────────────────────
        self.state     = x_next
        self.step_num += 1

        terminated = bool(np.abs(x_next[0]) > self.x_limit)
        truncated  = bool(self.step_num >= self.max_steps)

        info = {
            "stage_cost" : stage_cost,
            "step"       : self.step_num,
            "state"      : x_next.copy(),
        }

        return x_next.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        pos, vel = self.state
        print(f"  step {self.step_num:3d} | pos={pos:+8.4f} m  "
              f"vel={vel:+8.4f} m/s")

    # ── Simulation utility ─────────────────────────────────────────────────

    def simulate(self, controller_fn, x0=None):
        """
        Run one full episode using an arbitrary controller.

        Parameters
        ----------
        controller_fn : callable
            Takes obs (np.ndarray, shape (2,)) → action (np.ndarray, shape (1,))
            Both lqr_controller() and ppo_controller() conform to this signature.
        x0 : array-like or None
            Override initial state. Defaults to [5.0, 0.0].

        Returns
        -------
        dict with keys:
            states      : (T+1, 2) — full state trajectory including x0
            positions   : (T+1,)
            velocities  : (T+1,)
            actions     : (T,)    — control inputs applied
            rewards     : (T,)
            stage_costs : (T,)
            total_cost  : float   — Σ stage_costs  (lower is better)
            n_steps     : int     — steps before termination or truncation

        Both controllers call this method → identical simulation machinery.
        """
        obs, _ = self.reset()

        # Override initial state if provided
        if x0 is not None:
            self.state = np.array(x0, dtype=np.float64)
            obs        = self.state.astype(np.float32)

        # Seed collections with x0
        states      = [self.state.copy()]
        actions     = []
        rewards     = []
        stage_costs = []

        done = False
        while not done:
            action = controller_fn(obs)

            obs, reward, terminated, truncated, info = self.step(action)

            states.append(info["state"].copy())
            actions.append(float(np.atleast_1d(action)[0]))
            rewards.append(reward)
            stage_costs.append(info["stage_cost"])

            done = terminated or truncated

        states = np.array(states)  # (T+1, 2)

        return {
            "states"      : states,
            "positions"   : states[:, 0],
            "velocities"  : states[:, 1],
            "actions"     : np.array(actions),
            "rewards"     : np.array(rewards),
            "stage_costs" : np.array(stage_costs),
            "total_cost"  : float(np.sum(stage_costs)),
            "n_steps"     : len(actions),
        }