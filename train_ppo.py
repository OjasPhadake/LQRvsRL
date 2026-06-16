# train_ppo.py
#
# Entry point: train the PPO agent on the double integrator.
#
# Run:
#   python train_ppo.py               # train 100k steps (default)
#   python train_ppo.py --steps 50000 # shorter run for quick testing
#   python train_ppo.py --eval-only   # skip training, evaluate saved model
#
# Output:
#   results/data/ppo_model.zip            — saved SB3 model
#   results/data/ppo_results.npy          — trajectory dict
#   results/data/ppo_learning_curve.npy   — (timesteps, mean_rewards, std)
#   results/figures/ppo_trajectory.png
#   results/figures/ppo_learning_curve.png

import argparse
from controllers.rl_agent import run_ppo, evaluate_ppo
from controllers.lqr import settling_time
from analysis.plots import plot_trajectory, plot_learning_curve, C_PPO
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO on the double integrator."
    )
    parser.add_argument(
        "--steps", type=int, default=100_000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load and evaluate existing model."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.eval_only:
        # Load saved model and evaluate — useful for inspecting a trained agent
        # without re-running the full training loop
        print("\n  Loading saved model from results/data/ppo_model.zip ...")
        results = evaluate_ppo("results/data/ppo_model")

        t_settle = settling_time(results["positions"])
        print(f"\n  Steps : {results['n_steps']}")
        print(f"  Cost  : {results['total_cost']:.4f}")
        print(f"  Settle: {t_settle:.2f} s")

        plot_trajectory(
            results,
            title     = r"PPO Controller — Double Integrator ($x_0=[5,0]$)",
            color     = C_PPO,
            save_path = "results/figures/ppo_trajectory.png",
        )
    else:
        # Full training + evaluation pipeline
        run_ppo(total_timesteps=args.steps)