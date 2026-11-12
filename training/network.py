"""
The Q-network: maps a board to 4 Q-values, one per action.

Responsibilities:
  - Define the architecture (a torch.nn.Module).
  - Nothing else. Training logic lives in agent.py.

Architecture (decided, see docs/SPEC.md -> Network):
  - Input: [N, 16] integer exponents from encoding.py.
  - First step of forward(): one-hot each cell into 16 channels and flatten
    to [N, 256], so index 16 x cell + exponent is 1. F.one_hot needs int64
    input, so cast the stored uint8 boards first.
  - MLP 256 -> 256 -> 256 -> 4. ReLU after each hidden layer, nothing after
    the output.
  - Linear and ReLU only: no BatchNorm, dropout or LayerNorm. Every layer
    used here has to be rewritten by hand in web/js/agent.js.

Why the one-hot lives here and not in encoding.py: the replay buffer and every
batch stay as 16-byte boards, and the JS side never builds the 256-wide vector
at all. It computes the first layer as a lookup (docs/SPEC.md -> Model file).
Worth being able to explain: a one-hot followed by a Linear layer is the same
as an embedding lookup.

Hidden width comes from config.py. Go from 256 to 512 only if training stalls:
the model ships to every visitor's browser, so size matters.

Optional upgrade later: a dueling head (separate value and advantage streams).
Keep each stream a single Linear on the last hidden layer, so export.py can
fold it into one Linear and the JS side doesn't change.
"""
