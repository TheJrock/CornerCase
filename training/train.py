"""
Entry point: runs the training loop.

Usage (once implemented):  python train.py

Responsibilities:
  - Build the game, agent, and replay buffer from config.
  - Loop: observe -> act -> step -> store transition -> (after warm-up) sample
    a batch and learn -> periodically sync the target network.
  - Decay epsilon over time.
  - Log progress: episode score, max tile, loss, epsilon (print, CSV, or
    TensorBoard writing to training/runs/).
  - Periodically run a greedy evaluation and save checkpoints to
    training/checkpoints/.

Tip: keep checkpoints from a few stages of training (early / mid / late).
Exported side by side, they make a great "watch it learn" feature on the site.

Expect a full run to take hours, not minutes. Check that the loop works end to
end with a tiny config first.
"""
