# LQR vs Reinforcement Learning — Double Integrator

A comparative study of classical optimal control and model-free reinforcement
learning on a discrete-time double integrator system.

## Problem

Control a double integrator from $x_0 = [5, 0]$ to $x^* = [0, 0]$, minimising:

$$J = \sum_{k=0}^{\infty} \left( x_k^\top Q x_k + u_k^\top R u_k \right)$$

**Two solvers. Same objective. Different assumptions.**

| | LQR | PPO |
|---|---|---|
| Knows dynamics? | Yes (uses A, B) | No |
| Solves | Riccati equation (DARE) | Clipped surrogate objective |
| Policy | Linear: $u = -Kx$ | Neural network: $\pi_\theta(a\|s)$ |
| Optimality | Provably optimal (for LQ) | Near-optimal (empirical) |

## System

$$x_{k+1} = Ax_k + Bu_k, \quad A = \begin{bmatrix}1 & 0.1 \\ 0 & 1\end{bmatrix}, \quad B = \begin{bmatrix}0 \\ 0.1\end{bmatrix}$$

State: $x = [\text{position},\ \text{velocity}]$,
Action: $u \in [-5, 5]$ (continuous scalar)

## Project Structure

```
LQRvsRL/
├── envs/
│   └── double_integrator_env.py   # Shared Gymnasium environment
├── controllers/
│   ├── lqr.py                     # DARE solver, K computation, simulation
│   └── rl_agent.py                # PPO training + evaluation
├── analysis/
│   ├── plots.py                   # All figure generation
│   └── compare.py                 # Metrics: cost, settling time, effort
├── report/
│   └── report.tex                 # LaTeX writeup
├── results/
│   ├── figures/                   # Saved plots
│   └── data/                      # Saved trajectories (numpy)
├── run_lqr.py                     # Explore dynamics + run LQR
├── train_ppo.py                   # Train PPO agent
├── main_compare.py                # Full comparison: LQR vs PPO
└── requirements.txt
```

## Run Order

```bash
# 1. Explore dynamics (no controller yet)
python run_lqr.py

# 2. Solve LQR and simulate
python run_lqr.py --lqr

# 3. Train PPO agent
python train_ppo.py

# 4. Full side-by-side comparison
python main_compare.py
```

## Key Insight

The RL reward is the negative LQR stage cost:

$$r_k = -\left(x_k^\top Q x_k + u_k^\top R u_k\right)$$

Maximising $\sum r_k$ is identical to minimising $J$.
Same optimisation problem — LQR solves it analytically via dynamic programming;
PPO approximates the solution through environment interaction.