"""
Board -> network input. The bridge between the game and the neural network.

Responsibilities:
  - Convert a board into the tensor the Q-network takes as input.
  - Produce the legal-action mask in a consistent format.

Why it gets its own file: web/js/encoding.js must produce exactly the same
numbers, in the same order, for the same board. Keeping this small and
isolated makes it easy to port and to compare the two.

Encodings to consider:
  - Raw log2 values, normalized (16 floats). Simple, but the network has a
    hard time learning that two cells "match".
  - One-hot per cell: 16 cells x K channels (K around 16-18, where the channel
    index is the tile's exponent). The standard choice. Works with an MLP
    (flattened) or a small CNN (4x4xK).

Decide the exact layout (cell order, channel order, shape) and write it down in
docs/SPEC.md.
"""
