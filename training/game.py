"""
The 2048 game engine: the environment the agent learns in.

This is the foundation of the project. If the rules here are wrong, the agent
learns the wrong game, and the web version (web/js/game.js) won't match what
it was trained on.

Responsibilities:
  - Hold the 4x4 board state and the score.
  - Apply a move (up/right/down/left): slide, merge, compute points gained.
  - Spawn a new tile after any move that changed the board.
  - Report which moves are legal (a move is legal if it changes the board).
  - Detect game over (no legal moves left).
  - Gym-style interface: reset() and step(action) -> (next_state, reward, done, info).

Decisions to make (record the answers in docs/SPEC.md):
  - Store tiles as values (2, 4, 8...) or exponents (1, 2, 3...)? Exponents
    are smaller and map directly onto a one-hot encoding.
  - Action index order. It must match the JS side exactly.
  - What is the reward? Merge score is the obvious choice. Alternatives
    include the log of the merge score, +1 per move survived, or bonuses for
    empty cells. This choice shapes the agent's behavior more than almost
    anything else.
  - What happens if the agent picks an illegal move? Mask it out so it never
    can, or penalize it.
  - Accept a seed / RNG so episodes are reproducible.

Hint: write one function that slides a single row left, then get the other
three directions by rotating / transposing / reversing the board. That leads to
far fewer bugs than four hand-written versions.

Speed matters: training plays millions of moves. Get it correct first (tests!),
then make it fast.
"""
