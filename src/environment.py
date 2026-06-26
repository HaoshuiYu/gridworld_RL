"""
Graph-navigation MDP for the planning-vs-learning gap study.

Design note (why a graph and not a hardcoded grid):
The agent andplanner work solely from the transition tensor P, so new topologies (cycle,
expander, hypercube) plug in by adding a constructor -- nothing else changes.
"""

from __future__ import annotations

import numpy as np


class GraphMDP:
    """Graph navigation task: reach `goal` in fewest steps, with noisy moves.

    State = current vertex. Action = move to a chosen neighbour. Each step costs 1 
    (TODO: might revise later as hyperparameter) until the goal is reached; 
    with probability `slip`, the move goes to a random neighbour instead.

    adjacency : neighbours of each vertex.   
    goal : destination vertex.
    edge: undirected and unweighted
    slip : probability a move is stochastic.

    NOTE: "directionality" is trivial to graphs as you can distort or spin it, only connectivities define a graph.
    So, it's the most relevant information and actually teh only infomration relevant for MDP.
    """

    def __init__(self, adjacency: list[list[int]], goal: int, slip: float = 0.0):
        if not 0.0 <= slip < 1.0:
            raise ValueError("slip must be in [0, 1)")
        self.adjacency = [list(nbrs) for nbrs in adjacency]
        self.n_states = len(adjacency)
        self.goal = goal
        self.slip = slip
        self.n_actions = max(len(nbrs) for nbrs in self.adjacency)
        self._build_transition_tensor()

    def _build_transition_tensor(self) -> None:
        """P[s, a, s'] = Pr(s' | s, a), for the bellman equation for the Q* optimality (not policy) version"""
        n_s, n_a = self.n_states, self.n_actions
        P = np.zeros((n_s, n_a, n_s), dtype=np.float64)

        for s in range(n_s):
            nbrs = self.adjacency[s]
            deg = len(nbrs)
            if s == self.goal:
                # Absorbing goal: every action stays put.
                P[s, :, s] = 1.0
                continue
            for a in range(n_a):
                # Out-of-range action at a low-degree vertex -> self-loop.
                intended = nbrs[a] if a < deg else s
                # Intended move with prob (1 - slip).
                P[s, a, intended] += 1.0 - self.slip
                # Slip: uniformly random neighbour with prob slip.
                if deg > 0:
                    for nb in nbrs:
                        P[s, a, nb] += self.slip / deg
                else:
                    P[s, a, s] += self.slip
        self.P = P


    def reset(self, rng: np.random.Generator) -> int:
        """Begin always at a non-goal vertex of the graph at random."""
        s = self.goal
        while s == self.goal:
            s = int(rng.integers(self.n_states)) # TODO: build seed for randomization
        return s

    def step(self, s: int, a: int, rng: np.random.Generator):
        """Sample one transition. Returns (next_state, reward, done)."""
        s_next = int(rng.choice(self.n_states, p=self.P[s, a])) # environmental response to a given env from state s
        done = s_next == self.goal
        reward = -1.0 # implied cost of moving, should consider as hyperparameter down the line for subsequent experiments
        return s_next, reward, done



# Note: the vision for the graph is to be 3D, but the 3rd dimension is trivial as long as we articulate a 2D connection
# EX: a cube has vert, horiz, depth connections. But we can compress vert into horiz and depth as long as the other
# node vert above this one stays connected to all the same nodes as they compress

# in practice, many graphs here will be considerably higher dimensional/complex but compression into 2D

def make_grid(rows: int, cols: int, goal=None, slip: float = 0.0) -> GraphMDP:
    """Construct a 4-connected grid as a GraphMDP via adjacency list.

    Vertices: v = r * cols + c. Edges encodes neighbor connected by 1 
    move. stored as adjacency list: adjacency[n] = [list of neighbors of n] for all N
    """
    def vid(r, c):
        return r * cols + c # graphical compression: 2D -> 1D

    n = rows * cols
    adjacency = [[] for _ in range(n)]
    for r in range(rows):
        for c in range(cols):
            v = vid(r, c) # nested loop on v to articulate each distinct vertex
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc # considering ways to articulate diagonality 
                if 0 <= nr < rows and 0 <= nc < cols:
                    adjacency[v].append(vid(nr, nc)) # abandon "direcitonality" connectivity is sufficient to encode all graph info
    if goal is None:
        goal = n - 1  # bottom-right corner by default
    return GraphMDP(adjacency, goal=goal, slip=slip)