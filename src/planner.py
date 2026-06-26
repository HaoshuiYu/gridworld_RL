"""
Planner: this is optimal Q*, taking the bellman equation. It's the exact ground 
truth for every Q*(s,a) via value iteration on the KNOWN transition tensor P.

While randomness exists through slipping, the answer is deterministic since the
bellman computation is exact, env is known, and the computation itself is an 
expectation equation at heart. It accounts for the randomness in environment.

NOTE: an average is present since random slip is resolved through an expectation,
the expectation itself is the average and it's present in final output where our 
Policy (or optimal) is adjusted by distribution of slip for it via expectation. 
Nothing more, dont trip...
"""

from __future__ import annotations
import numpy as np
from environment import GraphMDP


def value_iteration(env: GraphMDP, gamma: float = 1.0, tol: float = 1e-10,
                    max_iters: int = 100_000):
    """Solve the known MDP exactly -> optimal V*, Q*, and optimal.

    gamma = 1.0 (undiscounted): goal is for "reward" function to be as close to 0,
    i.e. minimizing cost of travel (recall that's -1)
    """
    # set up
    n_s, n_a = env.n_states, env.n_actions
    P = env.P
    V = np.zeros(n_s, dtype=np.float64)

    # reward table, is given then used to build and refine Q* through mlutiple passes
    R = np.full((n_s, n_a), -1.0, dtype=np.float64)
    R[env.goal, :] = 0.0

    for _ in range(max_iters): # bellman, derive Q* recursively until max_iters
        # avg paths due to random env effects according to P matrix, but deterministic since P is fixed in environment.py
        Q = R + gamma * np.einsum("ijk,k->ij", P, V) 
        V_new = Q.max(axis=1)        
        V_new[env.goal] = 0.0 # each move has cost -1, so goal is to get back to reward = 0 for cost bounded by[-1,0]

        # NOTE: V_new update isn't fixing estimation error, that's for learner. 
        # for all V_new[n] where n = # of states, each pass updates all states at once.
        # EX: each pass starts at 0, then does conditioning where V_new[5] = -1 + V[6], recall to math 130B, C where
        # it gets chained and V[6] = -1 + P(V[7])V[7] + P(V[8])V[8], and so on. Each pass, it gets updated relative 
        # to final vertex then V[n] is informed by V[m] for each pass and produces a more "complete" computation 

        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    Q = R + gamma * np.einsum("ijk,k->ij", P, V) # final Q* with converged V*, accurate global reward comp for all states
    Q[env.goal, :] = 0.0 # set row to all 0s, stop there akin to absorbing state in stochastic processes
    policy = Q.argmax(axis=1)
    return V, Q, policy


def policy_expected_steps(env: GraphMDP, policy: np.ndarray, gamma: float = 1.0,
                          tol: float = 1e-10, max_iters: int = 100_000):
    """Scoring function for an arbitrary (learner) case of a policy 
    i.e. after t passes, a particular state onto goal state. Exact expected 
    steps to goal for ANY policy.

    Helps produce comparison V* - V = gap. It's V not Q because V = max_aQ[s,a] 
    because we're taking the value of a particular state.
    """
    n_s = env.n_states
    P = env.P
    V = np.zeros(n_s, dtype=np.float64)
    r = np.full(n_s, -1.0)
    r[env.goal] = 0.0

    for _ in range(max_iters):
        # Follow the policy's chosen action at each state: P[s, policy[s], :]
        Ppi = P[np.arange(n_s), policy]   
        V_new = r + gamma * Ppi @ V
        V_new[env.goal] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V