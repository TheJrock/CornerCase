# Spec: the contract between training and the web demo

The agent is trained in Python and runs in JavaScript. If anything below differs between the two sides, the demo breaks silently: the agent still "plays", but it makes nonsense moves. Fill this in as you make decisions and treat it as the source of truth for both sides.

## Board
- Representation: exponents, the log₂ of the tile value (2 → 1, 4 → 2, …, 2048 → 11). Convert back with 2ᵉ only for display.
- Cell order when flattened: row-major, top-left first. Cell index = 4 × row + col.
- Empty cell value: 0.
- Largest tile: 32768 (exponent 15). The Python engine's lookup tables give each cell 4 bits, so a larger tile can't be looked up, and without a check it gives wrong moves with no error.
  - A move that would merge two 32768 tiles raises a `ValueError`. Nothing changes first: the board and score stay as they were.
  - Setting a board that holds an exponent above 15, or a negative value, raises a `ValueError` too.
  - Only the chosen move is checked. A board where some other move would make a 65536 tile is still playable.
  - The JS engine enforces the same limit, and the network encoding only needs to cover exponents 0 to 15.
  - Revisit if an agent starts reaching 32768 (likely only the n-tuple agent in roadmap step 9): widen cells to 5 bits, or fall back to an exact slow slide for boards above 32768.

## Actions
| Index | Move |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |

## Tile spawning
- Spawns in a uniformly random empty cell: a 2 with probability 0.9, a 4 with probability 0.1.
- Only after a move that changed the board.
- RNG algorithm: _which seedable PRNG? It must give the same numbers in Python and JS for the same seed_
- Draws per spawn: _how many random numbers each spawn uses, and how each one maps to a cell and to a 2 or 4_
- Empty-cell order for picking a cell: _row-major?_

## Reward
- Definition: the sum of the new tile's exponent over every merge in the move. Two tiles with exponent `e` merge into `e + 1` and add `e + 1` to the reward. A move with no merges gives 0.
- Used for training only. Agents are evaluated on the real game score and max tile.
- Illegal move handling: masked, never penalized. Illegal actions are never chosen when acting, including ε-greedy random picks, and the TD target's max over next-state Q-values covers legal actions only.
- Game over: no extra penalty. The TD target for the final transition is just its reward, with no bootstrap term.

## Network input
- Range: exponents 0 to 15 (see Board → Largest tile).
- Encoding: _e.g. one-hot per cell, K channels_
- Shape: _e.g. [1, 16 * K] for an MLP_
- dtype: float32

## Network output
- Shape: [1, 4], Q-values in the action-index order above.
- Move selection: argmax over **legal** actions only.

## Model file
- Format: _JSON weights or ONNX?_
- Location: `web/models/<name>`
- Layout (if JSON): _layer names, weight shapes, row- vs column-major_

## Golden test cases
Boards with known correct results. Check both the Python tests and the JS engine against these.

| Row before | After sliding left | Points gained | Reward |
|---|---|---|---|
| 2 2 2 2 | 4 4 0 0 | 8 | 4 (2 + 2) |
| 2 2 4 0 | 4 4 0 0 | 4 | 2 |
| 4 0 0 4 | 8 0 0 0 | 8 | 3 |
| 2 4 8 16 | 2 4 8 16 (not a legal move) | 0 | n/a (masked) |
| 0 0 0 2 | 2 0 0 0 | 0 | 0 |
| 2 2 2 0 | 4 2 0 0 | 4 | 2 |
| 4 2 2 0 | 4 4 0 0 | 4 | 2 |
| 2 2 4 4 | 4 8 0 0 | 12 | 5 (2 + 3) |
| 0 2 0 2 | 4 0 0 0 | 4 | 2 |
| 2 0 2 2 | 4 2 0 0 | 4 | 2 |
| 2048 2048 0 0 | 4096 0 0 0 | 4096 | 12 |
| 2 4 2 4 | 2 4 2 4 (not a legal move) | 0 | n/a (masked) |
| 8 0 0 0 | 8 0 0 0 (not a legal move) | 0 | n/a (masked) |
| 32768 32768 0 0 | refused: raises, board unchanged (see Board → Largest tile) | 0 | n/a |

_Add more as you find edge cases. Add the Q-value fixtures from `training/export.py` here too, or link to a fixture file next to the model, so you can check the JS forward pass._
