"""
The DQN agent: owns the networks and the learning rule.

Responsibilities:
  - Hold the online Q-network and the target network (a periodically synced copy).
  - act(state, legal_mask, epsilon): epsilon-greedy over *legal* actions only.
  - learn(batch): compute TD targets and take one gradient step.
  - sync_target(): copy online weights to the target network (or soft-update with tau).
  - save() / load() checkpoints.

The core update (worth working through by hand before you code it):
    target = r + gamma * max_a' Q_target(s', a') * (1 - done)
    loss   = Huber(Q_online(s, a), target)

Gotchas:
  - Mask illegal actions in the max over a' as well, not just when acting
    (set them to -inf before taking the max).
  - Don't let gradients flow through the target. Only Q_online(s, a) should
    be trained.
  - An upgrade worth trying once basic DQN works: Double DQN (pick a' with the
    online network, score it with the target network).
"""

import numpy as np
from game import CornerCaseEnv

class RandomAgent:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng(rng)

    def reset(self, rng=None):
        self.rng = np.random.default_rng(rng)

    def act(self, legal_action_mask, state):
        legal_actions = np.flatnonzero(legal_action_mask)
        return self.rng.choice(legal_actions)

class PatternHeuristicAgent:
    def __init__(self, pattern=None):
        self.action_count = 0
        self.pattern = pattern or [CornerCaseEnv.DOWN, CornerCaseEnv.LEFT, CornerCaseEnv.UP, CornerCaseEnv.RIGHT]

    def reset(self, rng=None):
        self.action_count = 0

    def act(self, legal_action_mask, state):
        legal_actions = np.flatnonzero(legal_action_mask)
        action_index = self.action_count % len(self.pattern)
        illegal_move_counter = 0
        while self.pattern[action_index] not in legal_actions:
            self.action_count += 1
            action_index = self.action_count % len(self.pattern)
            illegal_move_counter += 1
            if illegal_move_counter > len(self.pattern):
                raise RuntimeError("No legal actions available in the pattern")
        self.action_count += 1
        return self.pattern[action_index]

class CornerHeuristicAgent:
    def __init__(self):
        pass

    def reset(self, rng=None):
        pass

    def act(self, legal_action_mask, state):
        legal_actions = np.flatnonzero(legal_action_mask)
        for action in [CornerCaseEnv.DOWN, CornerCaseEnv.LEFT, CornerCaseEnv.RIGHT, CornerCaseEnv.UP]:
            if action in legal_actions:
                return action
