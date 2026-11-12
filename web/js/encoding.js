/*
  Board -> network input. A JavaScript port of training/encoding.py.

  It must produce exactly the same numbers, in exactly the same order, as the
  Python version.

  Output: the 16 tile exponents flattened row-major (cell index = 4 x row +
  col), 0 for empty. No one-hot: agent.js uses the exponents directly as
  lookup indices into the first layer (see docs/SPEC.md -> Network input).

  This file is small now, but a wrong cell order here still breaks the demo
  without any error message: the agent keeps "playing", just badly. The index
  examples in docs/SPEC.md and the Q-value fixtures from training/export.py
  both catch it.
*/
