"""
Tabular Q-learning agent with convergence analysis:

Robbins Monro is applied to the learning rate where sum alpha = infinity (so it's 
always possible to reach the answer AND sum alpha^2 = finite (forces var to be finite)
Then out of remaining available values for alpha, we run iterations to configure the
best one.
This is necessary since each sample of same Q[s,a] isn't fully independent, so we can't
simply invoke SLLN to denoise (bc updating Q[s,a] on each visit is still the same state
with similar dependencies as prior)

The performance of the learner is measured through the local policy optimal against the 
GT optimal without averaging since the average would include noisier actions and exploration
actions without high reward functions. Instead, we wish to use the best policy "computed path"

TLDR: "Prior actions are inferior to final Q since all of them include exploration noise." 
"""

from __future__ import annotations
import numpy as np
from environment import GraphMDP

class QLearningAgent:
    def __init__(self, env: GraphMDP, alpha_exponent: float = 0.7,
                 epsilon: float = 0.1, gamma: float = 1.0, seed: int = 0):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_exponent = alpha_exponent  # how to sample from visiting same state: low if noise high, vice versa 
        self.rng = np.random.default_rng(seed) # seeded resolves reproducibility on environment
        self.Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
        self.visits = np.zeros((env.n_states, env.n_actions), dtype=np.int64)

    def _alpha(self, s: int, a: int) -> float: # samples from revisiting same [s,a], TD estimate improve to denoise
        return self.visits[s, a] ** (-self.alpha_exponent) # Robbins Monroe, constrains available alphas then we choose

    def act(self, s: int) -> int:
        if self.rng.random() < self.epsilon: 
            return int(self.rng.integers(self.env.n_actions)) # random chance to explore new info, else choose max
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def train(self, n_steps: int, max_episode_len: int = 1000):
        env, rng = self.env, self.rng
        s = env.reset(rng)
        ep_len = 0
        for _ in range(n_steps): # act and reward post action
            a = self.act(s)
            self.visits[s, a] += 1
            s_next, r, done = env.step(s, a, rng)

            target = r + (0.0 if done else self.gamma * self.Q[s_next].max()) # declares goal and loop until goal
            self.Q[s, a] += self._alpha(s, a) * (target - self.Q[s, a]) #TD error

            ep_len += 1
            if done or ep_len >= max_episode_len: # capped at 1000 iterations so agent doesn't wonder endlessly
                s = env.reset(rng)
                ep_len = 0
            else:
                s = s_next

    # ease to call
    def greedy_policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)

    def min_visit_count(self) -> int:
        non_goal = [s for s in range(self.env.n_states) if s != self.env.goal]
        return int(self.visits[non_goal].min())

    def q_gap_to_optimum(self, Q_star: np.ndarray) -> float:
        return float(np.max(np.abs(self.Q - Q_star)))