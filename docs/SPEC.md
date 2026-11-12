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
- The board itself: 16 integer exponents (0 to 15, see Board → Largest tile), flattened in the cell order above. Shape [N, 16]. No other inputs, such as the score or the number of empty cells.
- `encoding.py` and `encoding.js` only flatten the board. They don't one-hot it.
- The one-hot happens inside the model, as the first step of the forward pass:
  - K = 16 channels per cell. Channel index = exponent, so channel 0 means empty.
  - Flat index = 16 × cell + exponent, giving [N, 256]. This is the order `F.one_hot(board, 16).flatten(1)` produces from an [N, 16] board.
  - `F.one_hot` needs int64 input, so cast stored uint8 boards before calling it.
- dtype: the one-hot and every layer after it are float32.

Examples of the flat index:

| Cell (row, col) | Tile | Exponent | Index that is 1 |
|---|---|---|---|
| 0 (0, 0) | empty | 0 | 0 |
| 0 (0, 0) | 2 | 1 | 1 |
| 1 (0, 1) | empty | 0 | 16 |
| 5 (1, 1) | 8 | 3 | 83 |
| 15 (3, 3) | 2048 | 11 | 251 |
| 15 (3, 3) | 32768 | 15 | 255 |

## Legal-action mask
- bool[4] in the action-index order above, true = legal. This is what `game.py` returns in `info["legal_action_mask"]`.

## Network
- MLP: 256 → 256 → 256 → 4. ReLU after each hidden layer, nothing after the output.
- Layer types: Linear and ReLU only. No BatchNorm (it behaves differently in training and evaluation), no dropout, no LayerNorm, so the JS forward pass needs exactly two operations.
- Size: 133k parameters, about 0.7 MB per model file as sent (see Model file). Go to 512-wide hidden layers (396k parameters, about 2.2 MB as sent) only if training stalls. The browser fetches only the checkpoints the page actually loads.
- Dueling head (optional): allowed only if the value and advantage streams are each a single Linear on the last hidden layer. Then Q = V + A − mean(A) is still linear in that layer, and `export.py` folds it into one [4, H] Linear, so the JS side doesn't change:
  - W_q[i] = w_v + W_a[i] − mean_j W_a[j]
  - b_q[i] = b_v + b_a[i] − mean_j b_a[j]

## Network output
- Shape: [1, 4], Q-values in the action-index order above.
- Move selection: argmax over **legal** actions only.

## Model file
- Runs on a hand-written JS forward pass, not ONNX.
- Format: one JSON file per checkpoint. The browser fetches it in the background when the page loads; visitors don't do anything. At 256 wide it's about 0.7 MB as sent (GitHub Pages compresses JSON in transit). A raw float32 `.bin` would be about 0.5 MB. Switch only if you ship several 512-wide checkpoints, and then only `export.py` and the loader in `agent.js` change.
- Location: `web/models/<name>.json`, for example `dqn-early.json`, `dqn-mid.json`, `dqn-late.json`.
- Numbers: 9 significant digits, the fewest that load back into float32 exactly. Never round further, or the browser's Q-values drift from PyTorch's. No NaN or infinity: JSON can't hold them.
- Structure:
  ```
  {
    "meta": {"name": "dqn-late", "training_moves": …, "eval_mean_score": …},
    "layers": [
      {"weight": {"shape": [256, 256], "data": […]}, "bias": {"shape": [256], "data": […]}},
      {"weight": {"shape": [256, 256], "data": […]}, "bias": {"shape": [256], "data": […]}},
      {"weight": {"shape": [4, 256],   "data": […]}, "bias": {"shape": [4],   "data": […]}}
    ],
    "fixtures": [
      {"board": [16 exponents, row-major], "q": [4 raw Q-values]}
    ]
  }
  ```
- `meta`: lets the site label each checkpoint for the "watch it learn" feature ("after 200k moves").
- `layers`: in order from input to output. ReLU after every layer except the last (see Network). `data` is a flat array.
- Layout: each weight is stored like PyTorch's `nn.Linear.weight`: shape [out, in], row-major, so y_i = Σ_j W[i][j] · x_j + b_i.
- First layer: W₁ is [H, 256], where H is the hidden width. Column 16 × cell + exponent belongs to that cell and exponent. Only 16 of the 256 inputs are 1, so JS can compute layer 1 as a lookup instead of a multiply: h = b₁ + the sum, over the 16 cells, of W₁'s column 16 × cell + board[cell]. The result is the same with about 16× less work.
- `fixtures`: a handful of boards, including some with empty cells and large tiles, with the PyTorch model's float32 Q-values for each. These are raw Q-values, before masking illegal moves. The JS forward pass must match them within a relative error of 1e-4. Keeping fixtures inside the model file means each checkpoint carries its own check and can't be paired with the wrong one.

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

_Add more as you find edge cases. The Q-value fixtures for checking the JS forward pass live inside each model file (see Model file)._
