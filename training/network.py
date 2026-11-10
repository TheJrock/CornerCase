"""
The Q-network: maps an encoded board to 4 Q-values, one per action.

Responsibilities:
  - Define the architecture (a torch.nn.Module).
  - Nothing else. Training logic lives in agent.py.

Decisions to make:
  - MLP vs. small CNN. An MLP is the easiest to run by hand in the browser
    (it's just matrix multiplies). A CNN with 2x1 / 1x2 kernels can spot
    adjacent matches better, but it's harder to port without a library.
  - Width and depth. Start small (a couple of hidden layers with a few hundred
    units each). The model ships to every visitor's browser, so size matters.
  - Optional upgrade later: a dueling head (separate value and advantage streams).

Keep the export in mind: every layer type you use here must also exist in
whatever runs it in web/js/agent.js.
"""
