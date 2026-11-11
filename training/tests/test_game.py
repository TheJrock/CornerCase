"""
Tests for the game engine. Write these before (or while) you write game.py.

Run from training/ with:  pytest

Boards are written in exponents, as in docs/SPEC.md: 0 is empty, 1 is a 2,
2 is a 4, 11 is a 2048. Action indices also come from the SPEC.

What these tests assume about CornerCaseEnv:
  - CornerCaseEnv(rng=...) takes the RNG used for spawning.
  - env.board is a 4x4 integer array, and env.score is the game score in points.
  - env.step(action) -> (next_state, reward, done, info). An illegal action
    leaves everything unchanged and gives a reward of 0.
  - env.legal_action_mask() returns 4 booleans in action-index order.
  - env.spawn_tile() adds one tile to a random empty cell.

The cases in LINE_CASES and ILLEGAL_LINES are also listed in docs/SPEC.md so
the JS engine gets checked against the same expectations.
"""
import numpy as np
import pytest

from game import CornerCaseEnv

# Action indices, from docs/SPEC.md.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}


# ---------------------------------------------------------------- helpers

def make_rng(seed):
    """The one place to change if you switch to the PRNG chosen in SPEC.md."""
    return np.random.default_rng(seed)


def make_env(rows, seed=0):
    """An env holding a hand-made board (in exponents) with a score of 0."""
    env = CornerCaseEnv(rng=make_rng(seed))
    env.set_board(np.array(rows, dtype=env.board.dtype))
    env.score = 0
    return env


def assert_slid_then_spawned(actual, expected_slid):
    """actual must equal expected_slid plus exactly one new tile (a 2 or a 4)
    in a cell that expected_slid leaves empty.

    Spawning is random, so a test can't predict the whole board after a legal
    move. It can predict everything except the one spawned tile.
    """
    expected_slid = np.asarray(expected_slid)
    changed = np.argwhere(actual != expected_slid)
    assert len(changed) == 1, (
        "expected the slid board plus one spawned tile.\n"
        f"expected before the spawn:\n{expected_slid}\ngot:\n{actual}"
    )
    r, c = changed[0]
    assert expected_slid[r, c] == 0, f"tile spawned on an occupied cell ({r}, {c})"
    assert actual[r, c] in (1, 2), f"spawned exponent {actual[r, c]}, expected 1 or 2"


LINE_INDEX = 2  # which row or column place_line uses; any of 0-3 works


def place_line(line, action):
    """A 4x4 board holding `line` in one row or column, oriented so that
    `action` slides the tiles toward line[0]."""
    board = np.zeros((4, 4), dtype=int)
    if action == LEFT:
        board[LINE_INDEX, :] = line
    elif action == RIGHT:
        board[LINE_INDEX, ::-1] = line
    elif action == UP:
        board[:, LINE_INDEX] = line
    else:  # DOWN
        board[::-1, LINE_INDEX] = line
    return board


# ---------------------------------------------------------------- sliding and merging

# (line before, line after, points gained, reward). Each line is written in the
# direction of travel: the first entry is the cell the tiles slide toward.
LINE_CASES = [
    # The golden cases from SPEC.md
    ([1, 1, 1, 1], [2, 2, 0, 0], 8, 4),          # pairs merge from the leading edge
    ([1, 1, 2, 0], [2, 2, 0, 0], 4, 2),          # a merged tile can't merge again
    ([2, 0, 0, 2], [3, 0, 0, 0], 8, 3),          # tiles slide past gaps before merging
    # New cases
    ([0, 0, 0, 1], [1, 0, 0, 0], 0, 0),          # slide with no merge: legal, no reward
    ([1, 1, 1, 0], [2, 1, 0, 0], 4, 2),          # three equal: the leading pair merges
    ([2, 1, 1, 0], [2, 2, 0, 0], 4, 2),          # merged tile doesn't merge with its new neighbour
    ([1, 1, 2, 2], [2, 3, 0, 0], 12, 5),         # two different merges in one line
    ([0, 1, 0, 1], [2, 0, 0, 0], 4, 2),          # gaps before and between
    ([1, 0, 1, 1], [2, 1, 0, 0], 4, 2),          # gap inside the merging pair
    ([11, 11, 0, 0], [12, 0, 0, 0], 4096, 12),   # big tiles: points are 2 ** exponent
]

ILLEGAL_LINES = [
    [1, 2, 3, 4],   # the golden case from SPEC.md
    [1, 2, 1, 2],   # full and alternating
    [3, 0, 0, 0],   # already against the wall
    [0, 0, 0, 0],   # empty
]


@pytest.mark.parametrize("action", ACTIONS, ids=ACTION_NAMES.get)
@pytest.mark.parametrize(
    "line, expected, points, reward", LINE_CASES, ids=[str(c[0]) for c in LINE_CASES]
)
def test_line_slides_and_merges(line, expected, points, reward, action):
    env = make_env(place_line(line, action))
    _, got_reward, _, _ = env.step(action)
    assert_slid_then_spawned(env.board, place_line(expected, action))
    assert env.score == points
    assert got_reward == reward


@pytest.mark.parametrize("action", ACTIONS, ids=ACTION_NAMES.get)
@pytest.mark.parametrize("line", ILLEGAL_LINES, ids=str)
def test_illegal_move_changes_nothing(line, action):
    env = make_env(place_line(line, action))
    before = env.board.copy()
    _, reward, _, _ = env.step(action)
    assert np.array_equal(env.board, before), "an illegal move changed the board or spawned a tile"
    assert env.score == 0
    assert reward == 0
    assert not env.legal_action_mask()[action]


# A whole board where every row and column does something different. The
# expected results were worked out by hand rather than with place_line, so
# they don't share any rotation logic with the engine.
FULL_BOARD = [
    [1, 1, 0, 2],
    [0, 2, 2, 2],
    [3, 0, 3, 0],
    [1, 0, 0, 1],
]

FULL_BOARD_CASES = [
    # (action, board before the spawn, points, reward)
    (LEFT,  [[2, 2, 0, 0], [3, 2, 0, 0], [4, 0, 0, 0], [2, 0, 0, 0]], 32, 11),
    (RIGHT, [[0, 0, 2, 2], [0, 0, 2, 3], [0, 0, 0, 4], [0, 0, 0, 2]], 32, 11),
    (UP,    [[1, 1, 2, 3], [3, 2, 3, 1], [1, 0, 0, 0], [0, 0, 0, 0]], 8, 3),
    (DOWN,  [[0, 0, 0, 0], [1, 0, 0, 0], [3, 1, 2, 3], [1, 2, 3, 1]], 8, 3),
]


@pytest.mark.parametrize(
    "action, expected, points, reward", FULL_BOARD_CASES,
    ids=[ACTION_NAMES[c[0]] for c in FULL_BOARD_CASES],
)
def test_full_board_moves(action, expected, points, reward):
    env = make_env(FULL_BOARD)
    _, got_reward, _, _ = env.step(action)
    assert_slid_then_spawned(env.board, expected)
    assert env.score == points
    assert got_reward == reward


def test_score_accumulates_across_moves():
    env = make_env(FULL_BOARD)
    env.score = 1000
    env.step(LEFT)
    assert env.score == 1032


# ---------------------------------------------------------------- spawning

def test_spawn_uses_empty_cells_after_the_slide():
    # Before the move, the empty cells are (0, 0) to (0, 2). After sliding left
    # they're (0, 1) to (0, 3). The other rows are full and can't merge.
    rows = [
        [0, 0, 0, 1],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ]
    expected = np.array(rows)
    expected[0] = [1, 0, 0, 0]
    for seed in range(20):
        env = make_env(rows, seed)
        env.step(LEFT)
        assert_slid_then_spawned(env.board, expected)


def test_spawn_fills_the_only_empty_cell():
    rows = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 0, 11],
        [12, 13, 14, 15],
    ]
    env = make_env(rows)
    env.spawn_tile()
    assert_slid_then_spawned(env.board, rows)  # the one change must be at (2, 2)


def test_spawn_is_a_4_about_10_percent_of_the_time():
    env = make_env(np.zeros((4, 4)))
    n = 10_000
    fours = 0
    for _ in range(n):
        env.board.fill(0)
        env.spawn_tile()
        tiles = env.board[env.board != 0]
        assert len(tiles) == 1
        fours += tiles[0] == 2
    assert 0.08 < fours / n < 0.12


def test_spawn_location_is_uniform_over_empty_cells():
    env = make_env(np.zeros((4, 4)))
    counts = np.zeros((4, 4), dtype=int)
    for _ in range(16_000):
        env.board.fill(0)
        env.spawn_tile()
        counts += env.board != 0
    # About 1,000 per cell expected. The bounds are roughly 5 standard deviations.
    assert counts.min() > 850 and counts.max() < 1150, f"spawn counts per cell:\n{counts}"


# ---------------------------------------------------------------- reset and returned states

def test_reset_starts_with_two_tiles_and_zero_score():
    env = make_env(np.full((4, 4), 5))
    env.score = 123
    state = env.reset()
    tiles = env.board[env.board != 0]
    assert len(tiles) == 2
    assert set(tiles.tolist()) <= {1, 2}
    assert env.score == 0
    assert np.array_equal(state, env.board)


def test_returned_states_are_copies():
    # If step() hands back env.board itself, every state stored in the replay
    # buffer is the same array and changes on every move.
    env = CornerCaseEnv(rng=make_rng(0))
    state = env.reset()
    state[0, 0] = 99
    assert env.board[0, 0] != 99

    env = make_env(FULL_BOARD)
    state, _, _, _ = env.step(LEFT)
    assert np.array_equal(state, env.board)
    state[0, 0] = 99
    assert env.board[0, 0] != 99


def test_two_envs_dont_share_a_board():
    # Needed for evaluate.py and for two agents playing side by side.
    a = CornerCaseEnv(rng=make_rng(1))
    b = CornerCaseEnv(rng=make_rng(2))
    assert a.board is not b.board
    b.board[0, 0] = 7  # no spawn or reset makes a 7, so this can only come from b
    assert a.board[0, 0] != 7


# ---------------------------------------------------------------- legal moves and game over

def test_legal_action_mask_is_a_mask_in_action_order():
    # Every tile is against the left wall and nothing can merge, so only
    # RIGHT changes the board.
    env = make_env([
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
    ])
    assert list(env.legal_action_mask()) == [False, False, False, True]


def test_full_board_with_a_vertical_pair_allows_only_up_and_down():
    env = make_env([
        [1, 2, 3, 4],
        [1, 5, 6, 7],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
    ])
    assert list(env.legal_action_mask()) == [True, True, False, False]
    assert not env.game_over()


def test_full_board_with_a_horizontal_pair_allows_only_left_and_right():
    env = make_env([
        [1, 1, 2, 3],
        [2, 5, 3, 4],
        [3, 6, 4, 5],
        [4, 7, 5, 6],
    ])
    assert list(env.legal_action_mask()) == [False, False, True, True]
    assert not env.game_over()


CHECKERBOARD = [
    [1, 2, 1, 2],
    [2, 1, 2, 1],
    [1, 2, 1, 2],
    [2, 1, 2, 1],
]


def test_full_board_with_no_merges_is_game_over():
    env = make_env(CHECKERBOARD)
    assert list(env.legal_action_mask()) == [False, False, False, False]
    assert env.game_over()


def test_board_with_an_empty_cell_is_not_game_over():
    rows = [row[:] for row in CHECKERBOARD]
    rows[3][3] = 0
    env = make_env(rows)
    assert not env.game_over()


def test_legal_action_mask_and_game_over_dont_change_anything():
    # A common bug: checking legality by actually running each move.
    env = make_env(FULL_BOARD)
    before = env.board.copy()
    env.legal_action_mask()
    env.game_over()
    assert np.array_equal(env.board, before)
    assert env.score == 0


def test_step_reports_done_when_the_move_ends_the_game():
    # Sliding left merges the two 16s into a 32, leaving (0, 3) empty. Whether
    # a 2 or a 4 spawns there, nothing can merge afterward.
    env = make_env([
        [4, 4, 6, 7],
        [6, 7, 8, 9],
        [7, 8, 9, 10],
        [8, 9, 10, 11],
    ])
    _, reward, done, _ = env.step(LEFT)
    assert reward == 5
    assert done
    assert env.game_over()


def test_step_reports_not_done_after_an_ordinary_move():
    env = make_env(FULL_BOARD)
    _, _, done, _ = env.step(LEFT)
    assert not done


# ---------------------------------------------------------------- reproducibility

def play_first_legal_move(env, max_moves=300):
    """Play the lowest-index legal move each turn. Returns every board seen and the final score."""
    boards = [env.board.copy()]
    for _ in range(max_moves):
        legal = env.legal_action_mask()
        if not any(legal):
            break
        env.step(int(np.flatnonzero(legal)[0]))
        boards.append(env.board.copy())
    return boards, env.score


def test_same_seed_gives_the_same_game():
    boards_a, score_a = play_first_legal_move(CornerCaseEnv(rng=make_rng(42)))
    boards_b, score_b = play_first_legal_move(CornerCaseEnv(rng=make_rng(42)))
    assert len(boards_a) == len(boards_b)
    assert all(np.array_equal(x, y) for x, y in zip(boards_a, boards_b))
    assert score_a == score_b
