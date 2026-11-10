"""
Hyperparameters and paths: one place to change settings without editing logic.

Why a separate file: you will tweak these constantly, and you want each
training run's settings saved alongside its checkpoint so you can reproduce
results later.

Things that typically live here:
  - learning rate, discount factor (gamma), batch size
  - replay buffer capacity, warm-up steps before learning starts
  - epsilon schedule (start, end, decay steps)
  - target network sync frequency (hard copy every N steps, or soft update tau)
  - total training steps/episodes, eval frequency, checkpoint frequency
  - random seed
  - paths: checkpoints/, runs/, ../web/models/

Options: a dataclass, a plain dict, or a YAML file plus a loader. A dataclass
gives you autocomplete and type hints with no extra dependencies.
"""
