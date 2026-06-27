"""
Run the planning-vs-learning comparison on a single grid and render the figure.

Produces:
1. Console report: exact optimal expected-steps vs the learned policy's exact
   expected-steps, plus convergence diagnostics (min visit count, ||Q - Q*||).
2. figures/learning_gap.png: the learned policy's suboptimality (regret) against
   the exact planning optimum as training budget grows, averaged over seeds with
   a min/max band. The optimum is the flat reference line at 0.

v1: ONE topology (grid), the gap made visible. The topology sweep that turns this
into the paper plugs in by looping make_grid -> other constructors.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import make_grid
from planner import value_iteration, policy_expected_steps
from agent import QLearningAgent

# hyperparameters
ROWS, COLS = 5, 5
SLIP = 0.1
EPSILON = 0.1
ALPHA_EXP = 0.7
SEEDS = list(range(8))
# budget of steps taken: chosen to double: account for fast paced learning early and slower later on
BUDGETS = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]


def mean_optimal_steps(V_star, goal): # TODO: possibly unreachable nodes generated, should guard in graph generation
    """ for optimal: value = (-) Average cost of path of states (exclude goal state)"""
    mask = np.ones(len(V_star), dtype=bool)
    mask[goal] = False
    return float(np.mean(-V_star[mask]))


def mean_policy_steps(env, V_pi):
    """for policy but ^^"""
    mask = np.ones(env.n_states, dtype=bool)
    mask[env.goal] = False
    return float(np.mean(-V_pi[mask]))


def run():
    # build grid to track state coord and coord of opt pathof states
    env = make_grid(ROWS, COLS, slip=SLIP)
    V_star, Q_star, opt_policy = value_iteration(env)
    opt_steps = mean_optimal_steps(V_star, env.goal)

    print("=" * 60)
    print(f"Grid {ROWS}x{COLS} | slip={SLIP} | {len(SEEDS)} seeds")
    print("=" * 60)
    print(f"Exact optimal mean steps-to-goal (ground truth): {opt_steps:.3f}")
    print("-" * 60)

    # regret[budget_index, seed] = learned mean steps - optimal mean steps
    regret = np.zeros((len(BUDGETS), len(SEEDS)))
    final_min_visits = []
    final_q_gaps = []

    for si, seed in enumerate(SEEDS): # rng seed for randomness in how RL trains not in slip or environment
        agent = QLearningAgent(env, alpha_exponent=ALPHA_EXP,
                               epsilon=EPSILON, seed=seed)
        trained = 0
        for bi, budget in enumerate(BUDGETS):
            agent.train(budget - trained)
            trained = budget
            V_pi = policy_expected_steps(env, agent.greedy_policy())
            learned_steps = mean_policy_steps(env, V_pi)
            regret[bi, si] = learned_steps - opt_steps
        final_min_visits.append(agent.min_visit_count())
        final_q_gaps.append(agent.q_gap_to_optimum(Q_star))

    mean_regret = regret.mean(axis=1)
    lo, hi = regret.min(axis=1), regret.max(axis=1)

    print(f"Convergence diagnostics at final budget ({BUDGETS[-1]} steps):")
    print(f"  min visit count over (s,a): {int(np.min(final_min_visits))} "
          f"(want large -> infinite-visitation condition)")
    print(f"  ||Q - Q*|| max-norm, mean over seeds: "
          f"{np.mean(final_q_gaps):.4f} (want near 0)")
    print(f"  mean residual regret: {mean_regret[-1]:.4f} steps")
    print("=" * 60)

    # visualization
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0.0, color="#444", lw=1.4, ls="--",
               label="planning optimum (value iteration)")
    ax.plot(BUDGETS, mean_regret, color="#c0392b", lw=2.2, marker="o",
            label="learned policy (Q-learning)")
    ax.fill_between(BUDGETS, lo, hi, color="#c0392b", alpha=0.15,
                    label="min-max over seeds")
    ax.set_xscale("log")
    ax.set_xlabel("training steps (log scale)")
    ax.set_ylabel("excess expected steps-to-goal\n(learned - optimal)")
    ax.set_title(f"Planning-vs-Learning Gap on a {ROWS}x{COLS} Grid (slip={SLIP})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    out_path = os.path.normpath(os.path.join(out_dir, "learning_gap.png"))
    fig.savefig(out_path, dpi=130)
    print(f"Figure written to {out_path}")
    return out_path


if __name__ == "__main__":
    run()