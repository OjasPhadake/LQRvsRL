# controllers/rl_agent.py
#
# PPO agent for the discrete-time double integrator.
#
# Design choices and why:
# ────────────────────────────────────────────────────────────────────────────
#
# 1. ACTION NORMALISATION
#    SB3's PPO uses a Gaussian policy. The mean and std are outputs of the
#    network, and the action is sampled from N(μ, σ). SB3 assumes the action
#    space is roughly [-1, 1] for numerical stability of the Gaussian.
#    If the action space is [-5, 5], the network must output means of ±5,
#    which pushes activations into saturation and slows convergence.
#    Fix: use a RescaleAction wrapper so PPO sees [-1, 1] internally, but
#    the environment receives the rescaled value in [-5, 5].
#
# 2. PARALLEL ENVIRONMENTS (n_envs=4)
#    SB3's PPO collects n_steps per environment before each update.
#    With n_envs=4 and n_steps=1024, each update uses 4096 timesteps.
#    This gives more diverse trajectory data per gradient step — important
#    for a problem where early episodes may terminate in very different ways
#    depending on how quickly the agent learns to stabilise.
#
# 3. NETWORK ARCHITECTURE
#    Default MlpPolicy uses [64, 64] hidden layers with Tanh activations.
#    For a 2D linear system this is considerable overkill — the optimal
#    policy is a linear function u = -Kx. But we keep the default so that
#    the comparison with LQR is honest: PPO is not given any structural
#    information about the problem.
#
# 4. REWARD SCALE
#    r_k = -(x'Qx + u'Ru)
#    At t=0: r_0 ≈ -25 (x=[5,0], Q=I, u≈0).
#    At goal: r_k ≈ 0.
#    Episode reward range: roughly [-11700 (random), -364 (LQR optimal)].
#    PPO is reward-scale sensitive. We do NOT normalise rewards here because
#    we want the reward values to be directly comparable to LQR costs.
#    If training is unstable, reward normalisation via VecNormalize would
#    be the first thing to try.
#
# 5. EVALUATION CALLBACK
#    Records (timestep, mean_reward, std_reward) every eval_freq steps.
#    This produces the learning curve shown in Phase 5. Using a separate
#    deterministic evaluation environment (not the training envs) avoids
#    conflating exploration noise with actual policy quality.

import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from envs.double_integrator_env import DoubleIntegratorEnv


# ── Learning curve callback ────────────────────────────────────────────────

class LearningCurveCallback(BaseCallback):
    """
    Records mean episode reward on a separate evaluation environment
    every `eval_freq` training timesteps.

    Attributes
    ----------
    timesteps   : list[int]   — training timesteps at each evaluation
    mean_rewards: list[float] — mean episode reward at each evaluation
    std_rewards : list[float] — std  episode reward at each evaluation

    These are passed to analysis.plots.plot_learning_curve() in Phase 5.
    """

    def __init__(self, eval_env, eval_freq=5_000,
                 n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env       = eval_env
        self.eval_freq      = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # Filled during training — retrieved afterwards
        self.timesteps    = []
        self.mean_rewards = []
        self.std_rewards  = []

        self._last_eval_step = 0

    def _on_step(self) -> bool:
        # num_timesteps is maintained by SB3 across all parallel envs
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps

            mean_r, std_r = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes = self.n_eval_episodes,
                deterministic   = True,
                warn            = False,
            )

            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(mean_r)
            self.std_rewards.append(std_r)

            if self.verbose >= 1:
                print(f"  [{self.num_timesteps:>7d} steps]  "
                      f"mean reward = {mean_r:+8.2f}  "
                      f"std = {std_r:.2f}")

        return True   # return False to stop training early


# ── Environment factory ────────────────────────────────────────────────────

def _make_env():
    """
    Factory function for a single training environment.

    Wraps DoubleIntegratorEnv with:
      - Monitor  : lets SB3 track episode rewards and lengths automatically
      - RescaleAction : remaps action space from [-5,5] to [-1,1]

    The RescaleAction wrapper is the right fix for the SB3 warning about
    non-normalised action spaces. The wrapper is transparent to our
    analysis code — env.simulate() bypasses SB3 and calls the unwrapped
    env directly.
    """
    env = DoubleIntegratorEnv(max_steps=200)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = Monitor(env)
    return env


def _make_eval_env():
    """
    Separate evaluation environment — deterministic, no exploration noise.

    Using a separate eval env (not one of the training VecEnvs) ensures
    evaluation episodes are independent of training state.
    """
    env = DoubleIntegratorEnv(max_steps=200)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = Monitor(env)
    return env


# ── Training ───────────────────────────────────────────────────────────────

def train_ppo(
    total_timesteps = 100_000,
    n_envs          = 4,
    n_steps         = 1024,
    batch_size      = 64,
    n_epochs        = 10,
    learning_rate   = 3e-4,
    clip_range      = 0.2,
    gae_lambda      = 0.95,
    gamma           = 0.99,
    eval_freq       = 5_000,
    n_eval_episodes = 10,
    save_path       = "results/data/ppo_model",
    seed            = 42,
    verbose         = 1,
):
    """
    Train a PPO agent on the double integrator.

    Parameters
    ----------
    total_timesteps : int
        Total environment steps. 100k is sufficient for this 2D problem.
        Collected across all n_envs in parallel.
    n_envs : int
        Number of parallel training environments. More envs = more diverse
        data per update = stabler gradients. 4 is a good default.
    n_steps : int
        Steps collected per environment before each PPO update.
        Total data per update = n_envs × n_steps = 4 × 1024 = 4096 steps.
    batch_size : int
        Mini-batch size for the gradient update inside each epoch.
        n_epochs × (n_envs × n_steps / batch_size) = gradient steps per update.
    n_epochs : int
        Number of passes through the collected data per update.
        This is the key PPO efficiency gain over vanilla policy gradient.
    learning_rate : float
        Adam learning rate. 3e-4 is the standard PPO default.
    clip_range : float
        ε in the PPO clipped objective. Limits policy change per update.
        0.2 means the probability ratio r_t stays in [0.8, 1.2].
    gae_lambda : float
        λ in Generalised Advantage Estimation. Controls bias-variance
        tradeoff: λ=0 → one-step TD (low var, high bias),
                  λ=1 → Monte Carlo (unbiased, high var).
        0.95 is the standard choice from the PPO paper.
    gamma : float
        Discount factor. 0.99 weights rewards 100 steps away at 0.99^100 ≈ 0.37.
        For a 200-step episode with dense rewards, 0.99 is appropriate.
    eval_freq : int
        Evaluate policy every this many *total* timesteps (across all envs).
    save_path : str
        Where to save the trained model. SB3 appends .zip automatically.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    model    : stable_baselines3.PPO  — the trained agent
    callback : LearningCurveCallback  — contains .timesteps, .mean_rewards
    """
    os.makedirs("results/data",    exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("\n" + "=" * 55)
    print("  PPO TRAINING — Double Integrator")
    print("=" * 55)
    print(f"  total_timesteps : {total_timesteps:,}")
    print(f"  n_envs          : {n_envs}")
    print(f"  n_steps / env   : {n_steps}  "
          f"(total per update: {n_envs * n_steps:,})")
    print(f"  batch_size      : {batch_size}")
    print(f"  n_epochs        : {n_epochs}")
    print(f"  learning_rate   : {learning_rate}")
    print(f"  clip_range  ε   : {clip_range}")
    print(f"  gae_lambda  λ   : {gae_lambda}")
    print(f"  gamma       γ   : {gamma}")
    print(f"  eval_freq       : every {eval_freq:,} steps")
    print(f"  seed            : {seed}")
    print()
    print("  [Reward reminder]")
    print("  r_k = -(x'Qx + u'Ru) — same cost as LQR, negated.")
    print("  At x0=[5,0]: r_0 ≈ -25.0")
    print("  At x=[0,0]:  r_k =   0.0")
    print(f"  LQR total reward: {-364.09:.2f}  ← PPO target")
    print()

    # ── Build environments ─────────────────────────────────────────────────
    # make_vec_env creates n_envs copies and runs them in a single process
    # (DummyVecEnv). For heavier environments, SubprocVecEnv would use
    # multiple processes — overkill for a 2D linear system.
    train_env = make_vec_env(_make_env, n_envs=n_envs, seed=seed)
    eval_env  = _make_eval_env()

    # ── Instantiate PPO ────────────────────────────────────────────────────
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        n_steps         = n_steps,
        batch_size      = batch_size,
        n_epochs        = n_epochs,
        learning_rate   = learning_rate,
        clip_range      = clip_range,
        gae_lambda      = gae_lambda,
        gamma           = gamma,
        ent_coef        = 0.0,    # no entropy bonus — we want deterministic
                                   # convergence, not exploration
        vf_coef         = 0.5,    # value loss weight in combined loss
        max_grad_norm   = 0.5,    # gradient clipping for stability
        normalize_advantage = True,
        policy_kwargs   = dict(
            net_arch = [64, 64],  # two hidden layers of 64 units (tanh)
        ),
        verbose         = 0,      # suppress SB3's own logging; we use callback
        seed            = seed,
    )

    # ── Callback ───────────────────────────────────────────────────────────
    callback = LearningCurveCallback(
        eval_env        = eval_env,
        eval_freq       = eval_freq,
        n_eval_episodes = n_eval_episodes,
        verbose         = 1,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("  Training ...")
    print(f"  {'Step':>10}  {'Mean Reward':>13}  {'Std':>8}")
    print("  " + "─" * 38)

    model.learn(
        total_timesteps = total_timesteps,
        callback        = callback,
        progress_bar    = False,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    model.save(save_path)
    print(f"\n  Model saved → {save_path}.zip")

    # Save learning curve data for offline plotting
    np.save("results/data/ppo_learning_curve.npy", {
        "timesteps"    : np.array(callback.timesteps),
        "mean_rewards" : np.array(callback.mean_rewards),
        "std_rewards"  : np.array(callback.std_rewards),
    })
    print("  Learning curve saved → results/data/ppo_learning_curve.npy")

    train_env.close()
    eval_env.close()

    return model, callback


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_ppo(model_path="results/data/ppo_model"):
    """
    Load a saved PPO model and run one evaluation episode on the raw
    (unwrapped) DoubleIntegratorEnv so results are directly comparable
    to LQR.

    Returns
    -------
    dict : simulation results from env.simulate()
    """
    from stable_baselines3 import PPO as _PPO

    # Reload on the wrapped env (needed for model compatibility)
    wrapped_eval = _make_eval_env()
    model = _PPO.load(model_path, env=wrapped_eval)

    # Build a controller_fn that uses the trained policy
    # deterministic=True → use the mean of the Gaussian policy (no sampling)
    # This is the fair comparison point with LQR, which is deterministic.
    def ppo_controller(obs):
        action, _ = model.predict(
            obs.astype(np.float32),
            deterministic=True,
        )
        # RescaleAction means model outputs in [-1,1]; rescale back to [-5,5]
        return np.atleast_1d(action).astype(np.float64) * 5.0

    # Simulate on the raw unwrapped environment — same as LQR simulation
    raw_env = DoubleIntegratorEnv(max_steps=200)
    results = raw_env.simulate(ppo_controller, x0=[5.0, 0.0])

    wrapped_eval.close()
    return results


# ── run_ppo: called by train_ppo.py ───────────────────────────────────────

def run_ppo(total_timesteps=100_000):
    """
    Full PPO pipeline: train → evaluate → print metrics → save results.

    Called by train_ppo.py.
    Also called by main_compare.py (which loads saved model instead of retraining).

    Returns
    -------
    results  : dict from env.simulate()  — trajectory on raw env
    model    : trained PPO model
    callback : LearningCurveCallback
    """
    from controllers.lqr import settling_time
    from analysis.plots  import plot_trajectory, plot_learning_curve, C_PPO

    # ── Train ──────────────────────────────────────────────────────────────
    model, callback = train_ppo(total_timesteps=total_timesteps)

    # ── Evaluate on raw environment ────────────────────────────────────────
    print("\n  Evaluating trained policy on raw environment ...")
    results = evaluate_ppo("results/data/ppo_model")

    # ── Metrics (same set as LQR for direct comparison) ───────────────────
    t_settle = settling_time(results["positions"])

    print("\n  PPO EVALUATION METRICS")
    print("  " + "─" * 45)
    print(f"    Steps to termination : {results['n_steps']}")
    print(f"    Final position       : {results['positions'][-1]:+.6f} m")
    print(f"    Final velocity       : {results['velocities'][-1]:+.6f} m/s")
    print(f"    Settling time (±0.1) : {t_settle:.2f} s")
    print(f"    Total cost  J        : {results['total_cost']:.4f}")
    print(f"    LQR total cost  J    : 364.09   ← baseline")
    print(f"    Suboptimality vs LQR : "
          f"{100*(results['total_cost']-364.09)/364.09:.2f}%")
    print(f"    Peak |u|             : {np.max(np.abs(results['actions'])):.4f}")
    print(f"    RMS  |u|             : "
          f"{np.sqrt(np.mean(results['actions']**2)):.4f}")
    print()

    # ── Save trajectory ───────────────────────────────────────────────────
    np.save("results/data/ppo_results.npy", results)
    print("  Trajectory saved → results/data/ppo_results.npy")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_trajectory(
        results,
        title     = r"PPO Controller — Double Integrator ($x_0=[5,0]$)",
        color     = C_PPO,
        save_path = "results/figures/ppo_trajectory.png",
    )

    plot_learning_curve(
        callback.timesteps,
        callback.mean_rewards,
        callback.std_rewards,
        save_path = "results/figures/ppo_learning_curve.png",
    )

    return results, model, callback