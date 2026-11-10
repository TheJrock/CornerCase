/*
  The trained agent, running in the browser.

  Responsibilities:
    - Load the exported model from models/ (fetch() for JSON, or an ONNX
      Runtime Web session for ONNX).
    - For a given board: encode it (encoding.js), run the network, and get 4 Q-values.
    - Mask illegal moves and pick the argmax.
    - Expose the Q-values too. Showing them on screen ("how confident is it in
      each move?") makes the demo much more interesting.

  If you exported JSON weights for an MLP, the forward pass is just:
      for each layer: x = W·x + b, then ReLU (except after the last layer)
  which you can write by hand with loops over Float32Arrays.

  Check it against the Q-value fixtures from training/export.py before trusting it.
*/
