# Spec: the contract between training and the web demo

The agent is trained in Python and runs in JavaScript. If anything below differs between the two sides, the demo breaks silently: the agent still "plays", but it makes nonsense moves. Fill this in as you make decisions and treat it as the source of truth for both sides.

## Board
- Representation: _values (2, 4, 8…) or exponents (1, 2, 3…)?_
- Cell order when flattened: _row-major, top-left first?_
- Empty cell value: _0?_

## Actions
| Index | Move |
|---|---|
| 0 | _?_ |
| 1 | _?_ |
| 2 | _?_ |
| 3 | _?_ |

## Tile spawning
- Spawns in a uniformly random empty cell: a 2 with probability 0.9, a 4 with probability 0.1.
- Only after a move that changed the board.

## Reward
- Definition: _?_
- Illegal move handling: _masked (never chosen) or penalized?_

## Network input
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

| Row before | After sliding left | Points gained |
|---|---|---|
| 2 2 2 2 | 4 4 0 0 | 8 |
| 2 2 4 0 | 4 4 0 0 | 4 |
| 4 0 0 4 | 8 0 0 0 | 8 |
| 2 4 8 16 | 2 4 8 16 (not a legal move) | 0 |

_Add more as you find edge cases. Add the Q-value fixtures from `training/export.py` here too, or link to a fixture file next to the model, so you can check the JS forward pass._
