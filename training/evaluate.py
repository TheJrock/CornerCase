"""
Measures how good an agent actually is.

Usage (once implemented):  python evaluate.py <checkpoint>

Responsibilities:
  - Load a checkpoint and play N games greedily (epsilon = 0).
  - Report mean and median score, the distribution of max tile reached (% of
    games reaching 512, 1024, 2048...), and average game length.

Build baselines first and compare against them:
  - Random agent (a random legal move each turn).
  - A simple heuristic, such as a fixed preference order like
    down > left > right > up. That's the "corner strategy" this repo is named for.
If your DQN can't beat the heuristic, something is wrong.
"""
