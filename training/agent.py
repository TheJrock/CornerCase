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
