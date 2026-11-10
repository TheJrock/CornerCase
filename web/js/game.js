/*
  2048 game engine: a JavaScript port of training/game.py.

  It must behave exactly like the Python version: same action indices, same
  merge rules, same spawn probabilities. See docs/SPEC.md.

  Responsibilities:
    - Board state and score.
    - move(action): slide and merge, then return whether the board changed
      and how many points were gained.
    - Spawn a tile after a move that changed the board.
    - legalMoves() and isGameOver().

  No DOM code here. Keeping the rules separate from the drawing means you can
  test this file with node and reuse it for a mode where a human plays.

  Check it against the golden test cases in docs/SPEC.md.
*/
