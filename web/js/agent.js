/*
  The trained agent, running in the browser.

  Responsibilities:
    - Fetch models/<name>.json (format in docs/SPEC.md -> Model file) and copy
      each layer's flat "data" array into a Float32Array once, at load.
    - Check the fixtures stored in the model file on load: a few forward
      passes, compared within a relative error of 1e-4. Log a console warning
      on a mismatch, so a bad export or a forward-pass bug shows up at once.
    - For a given board: encode it (encoding.js), run the network, and get 4 Q-values.
    - Mask illegal moves and pick the argmax.
    - Expose the Q-values too. Showing them on screen ("how confident is it in
      each move?") makes the demo much more interesting.

  The forward pass is hand-written, with no libraries:
    - Layer 1 is a lookup, not a multiply. The network was trained on a
      one-hot of the board, so only 16 of its 256 inputs are 1. Start from the
      bias and add W1's column 16 x cell + exponent for each of the 16 cells.
      The result is the same as the full multiply, with about 16 times less work.
    - Every later layer: x = W·x + b, with W stored [out][in] row-major.
    - ReLU after every layer except the last.
  Loops over Float32Arrays are plenty fast for a network this size.

  Don't trust it until it passes the fixtures from training/export.py.
*/
