"""
Experience replay: a fixed-size memory of past transitions.

Each transition is (state, action, reward, afterstate, next_state, done), plus
the legal-action mask for next_state, which you'll need when computing targets.
The afterstate is the board after the slide and before the spawn
(info["afterstate"] from game.py). Plain DQN doesn't use it, but roadmap step
10 does, and storing it from the start means the buffer never has to change.

Responsibilities:
  - add(state, action, reward, afterstate, next_state, done, legal_action_mask):
    store one transition, overwriting the oldest when full (a circular
    buffer). A 4x4 board won't fit a 16-wide row. For now add() flattens any
    board that isn't already shape (16,). That's a placeholder: once
    encoding.py exists, train.py flattens boards with it and the check here
    goes. The mask is info["legal_action_mask"] from the step that produced
    next_state.
  - sample(batch_size): return a Batch of tensors on the buffer's device
    (config.py). Sample with replacement: draw all batch_size indices with one
    rng.integers call and index every field's array with them. Keep one
    Generator on the buffer, seeded from config, instead of building a new
    one on every call. Only call it after warm-up, since an empty buffer
    raises ValueError.
  - __len__: so training can wait until there are enough samples to learn from.

Why DQN needs it: consecutive moves are strongly correlated. Sampling randomly
from a large history breaks that correlation and lets each experience be
reused many times.

Storage: preallocate one numpy array per field instead of storing a Python
list of tuples. Store boards as raw exponents, uint8 [capacity, 16], for each
of state, afterstate and next_state. The network does the one-hot on each
sampled batch, so the buffer never holds encoded boards: 1M transitions take
about 48 MB as raw boards, against about 3 GB encoded. Raw boards are also
what step 10's "expected" target needs, since it lists every possible spawn
on the afterstate.

Speed, measured on the M1 Pro (2026-11-11) at batch 512 with 100k transitions:
    deque of tuples, one transition per sample() call:   ~6.9 ms per batch
    same, sharing one Generator:                         ~2.5 ms per batch
    numpy arrays, one vectorized index draw:             ~0.04 ms per batch
A learning step takes about 2.3 ms, so sampling one transition at a time would
make training 2 to 4 times slower. The deque also used about 10 times the
memory (63 MB against 6.5 MB at 100k transitions).

Optional upgrade later: prioritized replay (sample surprising transitions more often).
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    afterstates: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    next_legal_action_masks: torch.Tensor

class ReplayBuffer:
    def __init__(self, capacity=100_000, rng=None, device='cpu'):
        self.states = np.zeros((capacity, 16), dtype=np.uint8)
        self.afterstates = np.zeros((capacity, 16), dtype=np.uint8)
        self.next_states = np.zeros((capacity, 16), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool)
        self.legal_action_masks = np.zeros((capacity, 4), dtype=np.bool)
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(rng)
        self.device = device

    def __len__(self):
        return self.size

    def add(self, state, action, reward, afterstate, next_state, done, legal_action_mask):
        if state.shape != (16,):
            state = state.flatten()
        if afterstate.shape != (16,):
            afterstate = afterstate.flatten()
        if next_state.shape != (16,):
            next_state = next_state.flatten()
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.afterstates[self.position] = afterstate
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.legal_action_masks[self.position] = legal_action_mask
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        index = self.rng.integers(0, self.size, size=batch_size)
        return Batch(
            states=torch.tensor(self.states[index], device=self.device),
            actions=torch.tensor(self.actions[index], device=self.device),
            rewards=torch.tensor(self.rewards[index], device=self.device),
            afterstates=torch.tensor(self.afterstates[index], device=self.device),
            next_states=torch.tensor(self.next_states[index], device=self.device),
            dones=torch.tensor(self.dones[index], device=self.device),
            next_legal_action_masks=torch.tensor(self.legal_action_masks[index], device=self.device)
        )
