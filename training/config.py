"""
Hyperparameters and paths: one place to change settings without editing logic.

Why a separate file: you will tweak these constantly, and you want each
training run's settings saved alongside its checkpoint so you can reproduce
results later.

Things that typically live here:
  - learning rate, discount factor (gamma), batch size
  - replay buffer capacity, warm-up transitions before learning starts
    (compared in roadmap step 11, so keep it a plain setting)
  - how many game moves to play per learning step
  - epsilon schedule: start, end, and the fraction of the run to decay over.
    The baseline is fixed at 0.1 (start = end = 0.1, 10% random moves), since
    tile spawns already push the agent into varied boards. Roadmap step 12
    compares it with a linear decay from 0.99 to 0.1 over the first 10% of
    the run's environment steps, then 0.1 for the rest:
        epsilon = start + (end - start) * min(1, step / (0.1 * total_steps))
    With start = end the formula stays at 0.1, so one setting covers both
    arms. Store the 10% as a fraction rather than a step count, so the tiny
    test config decays over its own short run
  - target network sync frequency (hard copy every N steps, or soft update tau)
  - total training steps/episodes, eval frequency, checkpoint frequency
  - random seed, and the fixed set of evaluation seeds
  - device: "cpu" or "mps"
  - hidden layer width (256, see docs/SPEC.md -> Network)
  - paths: checkpoints/, runs/, ../web/models/
  - later, for roadmap step 10: target_mode ("sampled", "expected", "afterstate")

Device: default to "cpu". Benchmarked on the M1 Pro (torch 2.15, 2026-11-11)
with the 256 -> 256 -> 256 -> 4 network:
    choosing a move, batch 1:               CPU ~30,000/s   MPS ~2,000/s
    learning step, batch 512:               CPU ~430/s      MPS ~290/s
    learning step, batch 2048, 512 wide:    about even
Every GPU call has a fixed cost (starting the work, copying results back,
waiting for it to finish) that a network this small can't make up for.
Benchmark again if the network grows a lot.

Speed: the engine plays about 33,000 moves/s, so learning steps limit training
speed. The number of moves per learning step mostly decides how long a run takes.

Options: a dataclass, a plain dict, or a YAML file plus a loader. A dataclass
gives you autocomplete and type hints with no extra dependencies.
"""
