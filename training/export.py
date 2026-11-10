"""
Converts a trained checkpoint into a file the browser can load.

Usage (once implemented):  python export.py <checkpoint>

Writes to web/models/. That file is the only training output that ships to the site.

Pick one approach (web/js/agent.js is the other half):
  - JSON weights: write out each layer's weight matrix and bias as nested
    lists, and pair it with a hand-written forward pass in JS. No
    dependencies. The best fit for an MLP.
  - ONNX: torch.onnx.export(...), paired with ONNX Runtime Web in the browser.
    Use this if your architecture is too complex to port by hand.

Verify the export: run the same few boards through the PyTorch model and the
exported model, and confirm the Q-values match (within float tolerance). Save
those boards and expected outputs as a fixture so you can check the JS side too.
"""
