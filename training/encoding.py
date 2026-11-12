"""
Board -> network input. The bridge between the game and the neural network.

Responsibilities:
  - Turn a board, or a stack of boards, into the network's input: integer
    exponents with shape [N, 16], flattened row-major (cell index = 4 x row +
    col), 0 for empty. A single board is just N = 1.
  - Nothing more. The one-hot happens inside the model (network.py), so this
    file stays tiny.

Why it gets its own file: web/js/encoding.js must produce exactly the same
numbers, in the same order, for the same board. Keeping this small and
isolated makes it easy to port and to compare the two.

Where it's used: when acting, and when storing boards in the replay buffer
(as uint8, see replay_buffer.py).

The legal-action mask needs no encoding: game.py already returns bool[4] in
action-index order, which is the format in docs/SPEC.md.

Layout and dtype: docs/SPEC.md -> Network input.
"""
