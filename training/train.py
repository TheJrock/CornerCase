"""
Entry point: runs the training loop.

Usage (once implemented):  python train.py

Responsibilities:
  - Build the game, agent, and replay buffer from config, on the device that
    config.py names.
  - Loop: observe -> act -> step -> store transition -> (after warm-up, every
    few moves) sample a batch and learn -> periodically sync the target network.
  - Store the afterstate with each transition. game.py returns it in
    info["afterstate"], and replay_buffer.py explains why it's kept.
  - Act epsilon-greedy from the very first move, warm-up included, using the
    epsilon schedule in config.py (baseline: fixed 0.1). Count the schedule
    in environment steps from the start of the run.
  - Log progress: episode score, max tile, loss, and epsilon in runs where
    it decays (print, CSV, or TensorBoard writing to training/runs/).
  - Periodically run a greedy evaluation on one fixed set of seeds, so runs
    can be compared fairly (roadmap step 10 depends on this).
  - Save checkpoints to training/checkpoints/, together with the config that
    produced them.

Tip: keep checkpoints from a few stages of training (early / mid / late).
Exported side by side, they make a great "watch it learn" feature on the site.

Speed: learning steps set the pace, not the game (about 430 learning steps/s
at batch 512 on CPU, against about 33,000 game moves/s). How many moves you
play per learning step mostly decides how long a run takes.

Expect a full run to take hours, not minutes. Check that the loop works end to
end with a tiny config first.
"""
