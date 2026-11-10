"""
Experience replay: a fixed-size memory of past transitions.

Each transition is (state, action, reward, next_state, done), plus the legal
action mask for next_state, which you'll need when computing targets.

Responsibilities:
  - add(transition): store it, overwriting the oldest when full (a circular buffer).
  - sample(batch_size): return a random batch as tensors.
  - __len__: so training can wait until there are enough samples to learn from.

Why DQN needs it: consecutive moves are strongly correlated. Sampling randomly
from a large history breaks that correlation and lets each experience be
reused many times.

Performance tip: preallocate one numpy array per field instead of storing a
Python list of tuples.

Optional upgrade later: prioritized replay (sample surprising transitions more often).
"""
