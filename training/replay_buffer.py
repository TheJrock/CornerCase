"""
Experience replay: a fixed-size memory of past transitions.

Each transition is (state, action, reward, afterstate, next_state, done), plus
the legal-action mask for next_state, which you'll need when computing targets.
The afterstate is the board after the slide and before the spawn
(info["afterstate"] from game.py). Plain DQN doesn't use it, but roadmap step
10 does, and storing it from the start means the buffer never has to change.

Responsibilities:
  - add(transition): store it, overwriting the oldest when full (a circular buffer).
  - sample(batch_size): return a random batch as tensors.
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

Optional upgrade later: prioritized replay (sample surprising transitions more often).
"""
