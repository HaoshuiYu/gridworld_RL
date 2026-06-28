"""
Verification suite.

The project's central claim -- "the learned policy is compared against the EXACT
optimum" -- is only as trustworthy as the planner. We take instances where we can compute
with alternative algorithms to inspect the level of accuracy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environment import make_grid                          # noqa: E402
from planner import value_iteration, policy_expected_steps  # noqa: E402
from agent import QLearningAgent                           # noqa: E402


def test_deterministic_line_optimal_steps():
    """On a 1xN line with no slip, optimal steps from vertex i is (N-1-i), it helps inspect value iteration"""
    env = make_grid(1, 6, slip=0.0)
    V, Q, pi = value_iteration(env)
    assert np.allclose(-V, [5, 4, 3, 2, 1, 0])


def test_policy_eval_matches_optimal_value():
    """Exact evaluation of the optimal policy must reproduce V*, it helps inspect scorer vs answer"""
    env = make_grid(3, 3, slip=0.1)
    V, Q, pi = value_iteration(env)
    V_pi = policy_expected_steps(env, pi)
    assert np.allclose(V_pi, V)


def test_qlearning_converges_under_slip():
    """Q-learning should reach near-zero regret on a small slippery grid, it helps inspect convergence of Q"""
    env = make_grid(4, 4, slip=0.3)
    V, Q, pi_star = value_iteration(env)
    opt = -V[:-1].mean()
    agent = QLearningAgent(env, seed=1)
    agent.train(120_000)
    V_pi = policy_expected_steps(env, agent.greedy_policy())
    regret = (-V_pi[:-1].mean()) - opt
    assert regret < 0.05
    assert agent.min_visit_count() > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All checks passed.")
