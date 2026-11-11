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

import numpy as np

MAX_EXPONENT = 15

LEFT_ROWS = np.zeros(((MAX_EXPONENT + 1)**4, 4), dtype=int)
LEFT_POINTS = np.zeros((MAX_EXPONENT + 1)**4, dtype=int)
LEFT_REWARDS = np.zeros((MAX_EXPONENT + 1)**4, dtype=int)
LEFT_MOVED = np.zeros((MAX_EXPONENT + 1)**4, dtype=bool)
RIGHT_ROWS = np.zeros(((MAX_EXPONENT + 1)**4, 4), dtype=int)
RIGHT_POINTS = np.zeros((MAX_EXPONENT + 1)**4, dtype=int)
RIGHT_REWARDS = np.zeros((MAX_EXPONENT + 1)**4, dtype=int)
RIGHT_MOVED = np.zeros((MAX_EXPONENT + 1)**4, dtype=bool)
LEFT_TABLES = (LEFT_ROWS, LEFT_REWARDS, LEFT_MOVED, LEFT_POINTS)
RIGHT_TABLES = (RIGHT_ROWS, RIGHT_REWARDS, RIGHT_MOVED, RIGHT_POINTS)
WEIGHTS = np.array([1, (MAX_EXPONENT + 1), (MAX_EXPONENT + 1)**2, (MAX_EXPONENT + 1)**3])

class CornerCaseEnv():    

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.board = np.zeros((4, 4), dtype=int)
        self.cache = {}
        self.score = 0
        self.reset()

    def set_board(self, board):
        if board.shape != (4, 4):
            raise ValueError("Board must be a 4x4 array.")
        if np.any(board > MAX_EXPONENT):
            raise ValueError(f"Tile values greater than {2**MAX_EXPONENT:,} are not allowed.")
        if np.any(board < 0):
            raise ValueError("Tile values must be non-negative.")
        self.board = board.copy()
        self.build_cache()

    def build_cache(self):
        row_indeces = self.board @ WEIGHTS
        col_indeces = WEIGHTS @ self.board
        self.cache[self.LEFT] = self._action_lookup(row_indeces, LEFT_TABLES, False)
        self.cache[self.RIGHT] = self._action_lookup(row_indeces, RIGHT_TABLES, False)
        self.cache[self.UP] = self._action_lookup(col_indeces, LEFT_TABLES, True)
        self.cache[self.DOWN] = self._action_lookup(col_indeces, RIGHT_TABLES, True)

    @staticmethod
    def _action_lookup(indices, tables, transpose):
        board = tables[0][indices].T if transpose else tables[0][indices]
        reward = np.sum(tables[1][indices])
        moved = np.any(tables[2][indices])
        points = np.sum(tables[3][indices])
        return board, reward, moved, points

    def reset(self):
        self.board.fill(0)
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()
        self.build_cache()
        return self.board.copy()

    def step(self, action):
        info = {}
        next_board, reward, moved, points = self.cache[int(action)]
        if np.any(next_board > MAX_EXPONENT):
            raise ValueError(f"Merging two {2**MAX_EXPONENT:,} tiles would make a {2**(MAX_EXPONENT+1):,} tile, which the lookup tables can't hold.")
        self.board = next_board.copy()
        self.score += points
        afterstate = self.board.copy()
        if moved:
            self.spawn_tile()
            self.build_cache()
        mask = self.legal_action_mask()
        done = not any(mask)
        info["legal_actions"] = mask
        info["afterstate"] = afterstate
        return self.board.copy(), reward, done, info

    def legal_action_mask(self):
        moves = [False, False, False, False]
        for action in self.cache:
            _, _, moved, _ = self.cache[action]
            if moved:
                moves[action] = True
        return moves

    def game_over(self):
        return not any(self.legal_action_mask())

    def spawn_tile(self):
        empty_cells = np.flatnonzero(self.board == 0)
        if empty_cells.size > 0:
            idx = empty_cells[self.rng.integers(len(empty_cells))]
            self.board.flat[idx] = 1 if self.rng.random() < 0.9 else 2
        else:
            raise RuntimeError("No empty cells to spawn a new tile.")

    @staticmethod
    def _decode_row(encoded):
        row = np.zeros(4, dtype=int)
        for i in range(4):
            row[i] = (encoded // ((MAX_EXPONENT + 1) ** i)) % (MAX_EXPONENT + 1)
        return row

    @staticmethod
    def _slide_row_left(row):
        non_zero = row[row != 0]
        new_row = np.zeros_like(row)
        skip = False
        j = 0
        points = 0
        reward = 0
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                new_row[j] = non_zero[i] + 1
                points += 2 ** new_row[j]
                reward += new_row[j]
                skip = True
            else:
                new_row[j] = non_zero[i]
            j += 1
        return new_row, points, reward

def _build_lookup_tables():
    for i in range((MAX_EXPONENT + 1)**4):
        row = CornerCaseEnv._decode_row(i)
        left_row, left_points, left_reward = CornerCaseEnv._slide_row_left(row)
        right_row, right_points, right_reward = CornerCaseEnv._slide_row_left(row[::-1])
        right_row = right_row[::-1]
        LEFT_ROWS[i] = left_row
        LEFT_POINTS[i] = left_points
        LEFT_REWARDS[i] = left_reward
        LEFT_MOVED[i] = not np.array_equal(row, left_row)
        RIGHT_ROWS[i] = right_row
        RIGHT_POINTS[i] = right_points
        RIGHT_REWARDS[i] = right_reward
        RIGHT_MOVED[i] = not np.array_equal(row, right_row)

_build_lookup_tables()