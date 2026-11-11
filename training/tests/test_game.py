"""
Tests for the game engine. Write these before (or while) you write game.py.

Run from training/ with:  pytest

Boards are written in exponents, as in docs/SPEC.md: 0 is empty, 1 is a 2,
2 is a 4, 11 is a 2048. Action indices also come from the SPEC.

What these tests assume about CornerCaseEnv:
  - CornerCaseEnv(rng=...) takes the RNG used for spawning.
  - env.board is a 4x4 integer array, and env.score is the game score in points.
  - env.set_board(board) replaces the board, along with anything cached about it.
  - env.step(action) -> (next_state, reward, done, info). An illegal action
    leaves everything unchanged and gives a reward of 0. An action outside
    0-3 raises. info["afterstate"] is the board after the slide and before the
    spawn, and info["legal_actions"] is the legal-action mask for next_state.
  - A tile above 32768 (exponent 15) is refused with a ValueError: by step()
    for a move that would make one, and by set_board() for a board holding one.
    set_board() also refuses negative values with a ValueError.
  - env.legal_action_mask() returns 4 booleans in action-index order.
  - env.spawn_tile() adds one tile to a random empty cell.

The cases in LINE_CASES and ILLEGAL_LINES are also listed in docs/SPEC.md so
the JS engine gets checked against the same expectations.

The sections after "reproducibility" compare the engine against
reference_move, a slow plain-Python version of the rules that shares no code
with game.py, on thousands of boards. The lookup-table section reaches into
game.py's internals through a few accessor functions at its top, so renaming
or moving those internals only means changing the accessors.
"""
import copy
import functools

import numpy as np
import pytest

import game
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


# ---------------------------------------------------------------- reference implementation
#
# A slow, plain-Python version of the rules, written from SPEC.md. It uses no
# lookup tables, rotations or transposes, so it can't share a mistake with
# game.py. The tests below check the engine against it on thousands of boards,
# not just on the hand-written cases above.

def reference_slide_line(line):
    """Slide one line toward line[0]. Returns (new line, points, reward)."""
    tiles = [t for t in line if t != 0]
    new_line, points, reward = [], 0, 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] + 1
            new_line.append(merged)
            points += 2 ** merged
            reward += merged
            i += 2  # a merged tile can't merge again
        else:
            new_line.append(tiles[i])
            i += 1
    return new_line + [0] * (len(line) - len(new_line)), points, reward


def line_cells(action):
    """The four lines `action` slides, each a list of (row, col) cells in the
    direction of travel: the first cell is the one the tiles slide toward."""
    if action == LEFT:
        return [[(r, c) for c in range(4)] for r in range(4)]
    if action == RIGHT:
        return [[(r, c) for c in reversed(range(4))] for r in range(4)]
    if action == UP:
        return [[(r, c) for r in range(4)] for c in range(4)]
    return [[(r, c) for r in reversed(range(4))] for c in range(4)]  # DOWN


def reference_move(board, action):
    """(board after the slide, points, reward, moved). No spawn."""
    board = np.array(board)
    after = board.copy()
    points = reward = 0
    for cells in line_cells(action):
        line = [int(board[cell]) for cell in cells]
        new_line, line_points, line_reward = reference_slide_line(line)
        for cell, value in zip(cells, new_line):
            after[cell] = value
        points += line_points
        reward += line_reward
    return after, points, reward, not np.array_equal(after, board)


def reference_mask(board):
    return [reference_move(board, action)[3] for action in ACTIONS]


def first_legal_action(env):
    return int(np.flatnonzero(env.legal_action_mask())[0])


def step_and_check(env, action):
    """Take `action` and check everything step() reports against reference_move.
    Returns (next_state, done)."""
    before = env.board.copy()
    score_before = env.score
    after, points, reward, moved = reference_move(before, action)
    where = f"{ACTION_NAMES[action]} on\n{before}"

    state, got_reward, done, info = env.step(action)

    assert np.array_equal(info["afterstate"], after), (
        f"wrong afterstate after {where}\nexpected:\n{after}\ngot:\n{info['afterstate']}"
    )
    assert got_reward == reward, f"reward {got_reward}, expected {reward}, after {where}"
    assert env.score - score_before == points, (
        f"gained {env.score - score_before} points, expected {points}, after {where}"
    )
    if moved:
        assert_slid_then_spawned(state, after)
    else:
        assert np.array_equal(state, before), f"an illegal move changed the board: {where}"
    assert np.array_equal(state, env.board), "next_state isn't the env's board"

    expected_mask = reference_mask(state)
    assert list(info["legal_actions"]) == expected_mask, (
        f"info['legal_actions'] is wrong for\n{state}\nafter {where}"
    )
    assert list(env.legal_action_mask()) == expected_mask, (
        f"legal_action_mask() is wrong for\n{state}\nafter {where}"
    )
    assert done == (not any(expected_mask)), f"done is {done} for\n{state}"
    assert env.game_over() == done
    return state, done


@pytest.mark.parametrize(
    "line, expected, points, reward", LINE_CASES, ids=[str(c[0]) for c in LINE_CASES]
)
def test_reference_agrees_with_the_hand_written_lines(line, expected, points, reward):
    # Check the checker: if the reference were wrong, every test using it
    # would be checking the engine against the wrong rules.
    assert reference_slide_line(line) == (expected, points, reward)


@pytest.mark.parametrize(
    "action, expected, points, reward", FULL_BOARD_CASES,
    ids=[ACTION_NAMES[c[0]] for c in FULL_BOARD_CASES],
)
def test_reference_agrees_with_the_hand_written_full_board(action, expected, points, reward):
    after, got_points, got_reward, moved = reference_move(FULL_BOARD, action)
    assert np.array_equal(after, expected)
    assert (got_points, got_reward, moved) == (points, reward, True)


# ---------------------------------------------------------------- engine vs. reference

# (highest exponent, chance each cell is empty) for each kind of random board.
BOARD_KINDS = {
    "merge-heavy": (3, 0.3),   # small tiles, lots of equal neighbours
    "sparse": (6, 0.75),       # mostly empty: long slides past gaps
    "full": (8, 0.0),          # no empty cells: many illegal moves, some game overs
    "big tiles": (15, 0.3),    # the top of the 4-bit range, in every cell position
}


def random_boards(kind, n, seed):
    """n random boards of one kind. Skips boards where some move would make a
    16, which the engine refuses (see the last section of this file)."""
    highest, empty = BOARD_KINDS[kind]
    rng = np.random.default_rng(seed)
    boards = []
    while len(boards) < n:
        board = rng.integers(1, highest + 1, size=(4, 4))
        board[rng.random((4, 4)) < empty] = 0
        if any(reference_move(board, action)[0].max() > 15 for action in ACTIONS):
            continue
        boards.append(board)
    return boards


@pytest.mark.parametrize("kind", BOARD_KINDS)
def test_random_boards_match_the_reference(kind):
    # Every action from every board: legal mask, afterstate, reward, points,
    # next state and done. Catches a wrong table entry, a mixed-up direction
    # (rows vs. columns, the LEFT table where RIGHT belongs, a missing
    # transpose) and results reported for a different action than the one taken.
    for board in random_boards(kind, n=200, seed=list(BOARD_KINDS).index(kind)):
        env = make_env(board)
        assert list(env.legal_action_mask()) == reference_mask(board), (
            f"wrong legal mask for\n{board}"
        )
        for action in ACTIONS:
            step_and_check(make_env(board), action)


def test_long_random_games_match_the_reference():
    # Random actions, legal or not, through several whole games. Each step is
    # checked against the reference using the board the env actually holds,
    # so a cache that goes stale (not rebuilt after a spawn or a reset, or
    # reused after an illegal move when it shouldn't be) gets caught.
    env = CornerCaseEnv(rng=make_rng(7))
    pick = np.random.default_rng(8)
    games = 0
    assert list(env.legal_action_mask()) == reference_mask(env.board)
    for _ in range(3000):
        _, done = step_and_check(env, int(pick.integers(4)))
        if done:
            games += 1
            env.reset()
            assert list(env.legal_action_mask()) == reference_mask(env.board), (
                f"wrong legal mask right after reset() for\n{env.board}"
            )
    assert games >= 3, "the loop should have reached game over and reset several times"


# ---------------------------------------------------------------- what step() hands out

def test_results_handed_out_earlier_dont_change_later():
    # Anything step() or legal_action_mask() returns may be kept, e.g. in the
    # replay buffer. If it's the env's own array, or a view into the cache,
    # a later spawn, cache rebuild or reset rewrites it. An illegal move is
    # included because it skips the rebuild, which is when the env's board
    # and a cache entry can end up as the same array.
    env = make_env([
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
    ])
    kept = []

    def keep(label, value):
        kept.append((label, value, copy.deepcopy(value)))

    def play(action, label):
        state, _, _, info = env.step(action)
        keep(f"{label}: next_state", state)
        keep(f"{label}: info['afterstate']", info["afterstate"])
        keep(f"{label}: info['legal_actions']", info["legal_actions"])
        keep(f"{label}: legal_action_mask()", env.legal_action_mask())

    keep("legal_action_mask() before any move", env.legal_action_mask())
    play(LEFT, "illegal LEFT")   # no spawn, no rebuild
    play(RIGHT, "legal RIGHT")   # slide, spawn, rebuild
    for i in range(10):
        if env.game_over():
            break
        play(first_legal_action(env), f"move {i}")
    keep("reset() state", env.reset())
    play(first_legal_action(env), "first move after reset")
    env.reset()

    for label, value, snapshot in kept:
        assert np.array_equal(value, snapshot), f"{label} changed after it was returned"


def test_changing_what_step_returns_doesnt_change_the_env():
    # The other direction: the caller writing into a returned mask or array
    # must not reach the env's board or cache.
    env = make_env(FULL_BOARD)
    mask = env.legal_action_mask()
    mask[UP] = not mask[UP]
    assert list(env.legal_action_mask()) == reference_mask(FULL_BOARD)

    _, _, _, info = env.step(LEFT)
    board = env.board.copy()
    info["afterstate"][:] = 0
    info["legal_actions"][UP] = not info["legal_actions"][UP]
    assert np.array_equal(env.board, board)
    assert list(env.legal_action_mask()) == reference_mask(board)
    step_and_check(env, first_legal_action(env))  # the cache still works

    # After an illegal move nothing is rebuilt, so an afterstate that is
    # really a cache entry would carry the caller's write into the next move.
    env = make_env([
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
    ])
    _, _, _, info = env.step(LEFT)  # illegal
    info["afterstate"][:] = 0
    step_and_check(env, LEFT)       # still illegal, and still the same board
    step_and_check(env, RIGHT)


def test_two_envs_dont_share_a_cache():
    # A cache stored on the class (e.g. `cache = {}` or a preallocated array
    # in the class body) is shared by every env, so one env's moves change
    # another env's legal moves.
    a = make_env(CHECKERBOARD)
    b = make_env(FULL_BOARD)
    assert list(a.legal_action_mask()) == [False] * 4
    step_and_check(b, LEFT)
    assert list(a.legal_action_mask()) == [False] * 4


# ---------------------------------------------------------------- set_board and reset

def test_set_board_replaces_the_legal_moves():
    # set_board() must rebuild the cache, or the mask and the next move
    # describe the previous board.
    env = make_env(CHECKERBOARD)
    assert list(env.legal_action_mask()) == [False] * 4
    env.set_board(np.array(FULL_BOARD, dtype=env.board.dtype))
    assert list(env.legal_action_mask()) == reference_mask(FULL_BOARD)
    step_and_check(env, LEFT)


def test_set_board_keeps_its_own_copy():
    env = CornerCaseEnv(rng=make_rng(0))
    board = np.array(FULL_BOARD, dtype=env.board.dtype)
    env.set_board(board)
    board[:] = CHECKERBOARD  # the caller reuses their array
    assert np.array_equal(env.board, FULL_BOARD)
    assert list(env.legal_action_mask()) == reference_mask(FULL_BOARD)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (16,), (4, 4, 1)], ids=str)
def test_set_board_rejects_the_wrong_shape(shape):
    env = CornerCaseEnv(rng=make_rng(0))
    with pytest.raises(ValueError):
        env.set_board(np.zeros(shape, dtype=env.board.dtype))


# A negative cell makes a negative table index, and NumPy counts negative
# indices from the end of the table, so without the check the engine silently
# uses the wrong row. If the board's dtype ever becomes unsigned, negatives
# can't be stored at all and these can go.
BOARDS_WITH_A_NEGATIVE = {
    # Row 0 encodes to -1, which NumPy reads as the table's last entry: the
    # one for [15, 15, 15, 15].
    "-1 in a corner": [
        [-1, 0, 0, 0],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
    ],
    "-1 in the interior": [
        [1, 2, 3, 4],
        [2, -1, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ],
}


@pytest.mark.parametrize("board", BOARDS_WITH_A_NEGATIVE.values(), ids=list(BOARDS_WITH_A_NEGATIVE))
def test_set_board_refuses_a_negative_tile(board):
    env = make_env(FULL_BOARD)
    with pytest.raises(ValueError, match="negative"):
        env.set_board(np.array(board, dtype=env.board.dtype))
    # The env keeps the board it had, and its cache still matches it.
    assert np.array_equal(env.board, FULL_BOARD)
    assert list(env.legal_action_mask()) == reference_mask(FULL_BOARD)
    step_and_check(env, LEFT)


def test_reset_replaces_the_legal_moves():
    # reset() must rebuild the cache too, or the first move of every new game
    # is played on the last game's board.
    env = make_env(CHECKERBOARD)  # game over: nothing is legal
    env.reset()
    assert list(env.legal_action_mask()) == reference_mask(env.board)
    assert not env.game_over()  # two tiles on an empty board can always move
    step_and_check(env, first_legal_action(env))


# ---------------------------------------------------------------- actions

def test_action_constants_match_the_spec():
    # The JS side uses the same indices. ACTIONS at the top of this file is
    # the tests' own copy, so this checks the engine's.
    env_actions = (CornerCaseEnv.UP, CornerCaseEnv.DOWN, CornerCaseEnv.LEFT, CornerCaseEnv.RIGHT)
    assert env_actions == (UP, DOWN, LEFT, RIGHT) == (0, 1, 2, 3)


ACTION_TYPES = {
    "int": int,
    "numpy int64": np.int64,   # np.argmax, np.flatnonzero
    "numpy int32": np.int32,
    "0-d array": np.array,     # what q_values.argmax().numpy() gives
    "torch tensor": lambda a: pytest.importorskip("torch").tensor(a),  # q_values.argmax()
}


@pytest.mark.parametrize("convert", list(ACTION_TYPES.values()), ids=list(ACTION_TYPES))
def test_step_accepts_the_action_types_an_agent_produces(convert):
    # A cache in a dict only finds actions that hash like an int: a torch
    # tensor is a KeyError and a 0-d array isn't hashable at all.
    for action in ACTIONS:
        with_int = make_env(FULL_BOARD, seed=5)
        with_other = make_env(FULL_BOARD, seed=5)
        want = with_int.step(action)
        got = with_other.step(convert(action))
        assert np.array_equal(got[0], want[0]), ACTION_NAMES[action]
        assert got[1] == want[1], ACTION_NAMES[action]
        assert got[2] == want[2], ACTION_NAMES[action]
        assert with_other.score == with_int.score, ACTION_NAMES[action]


@pytest.mark.parametrize("action", [-1, 4, 10])
def test_step_rejects_an_action_outside_0_to_3(action):
    # With the cache in a dict, a bad action is a KeyError. With the cache in
    # arrays, -1 would quietly mean RIGHT (the last entry). Either way it must
    # raise, before changing anything.
    env = make_env(FULL_BOARD)
    before = env.board.copy()
    with pytest.raises((ValueError, IndexError, KeyError)):
        env.step(action)
    assert np.array_equal(env.board, before)
    assert env.score == 0


# ---------------------------------------------------------------- lookup tables
#
# These tests reach into game.py's internals. The accessors below are the
# only lines to change if you rename or move them, for example when merging
# the eight tables into one structure.

def encode_row(row):
    return CornerCaseEnv._encode_row(row)


def decode_row(index):
    return CornerCaseEnv._decode_row(index)


def table(direction):
    """(rows, points, rewards, moved) for sliding a row toward LEFT or RIGHT."""
    if direction == LEFT:
        return game.LEFT_ROWS, game.LEFT_POINTS, game.LEFT_REWARDS, game.LEFT_MOVED
    return game.RIGHT_ROWS, game.RIGHT_POINTS, game.RIGHT_REWARDS, game.RIGHT_MOVED


TABLE_FIELDS = ["rows", "points", "rewards", "moved"]
N_ROWS = 16 ** 4


def row_at(index):
    """The row a table index stands for: cell k is hex digit k, so cell 0 is
    the lowest digit. Written out here instead of calling decode_row, so a
    mistake in decode_row can't hide a mistake in a table."""
    return [(index >> (4 * k)) & 0xF for k in range(4)]


def test_encode_and_decode_example():
    # One hex digit per cell, cell 0 lowest: [1, 1, 2, 0] is 0x0211 = 529.
    assert encode_row([1, 1, 2, 0]) == 529
    assert list(decode_row(529)) == [1, 1, 2, 0]


def test_decode_then_encode_gives_back_every_index():
    bad = [i for i in range(N_ROWS) if encode_row(decode_row(i)) != i]
    assert not bad, f"{len(bad)} indices don't survive decode then encode, e.g. {bad[:5]}"


def test_decode_matches_the_table_numbering():
    for index in [0, 1, 0xF, 0x10, 529, 0x1234, 0xF00F, 0xFFFF]:
        assert list(decode_row(index)) == row_at(index), hex(index)


def test_encode_doesnt_overflow_on_the_largest_row():
    # 15*4096 + 15*256 + 15*16 + 15 = 65535. Narrowing the board or the
    # weights to a type like uint8 would overflow here.
    env = CornerCaseEnv(rng=make_rng(0))
    assert encode_row(np.full(4, 15, dtype=env.board.dtype)) == 0xFFFF


@functools.cache
def expected_table(direction):
    """What the table for `direction` should hold, from reference_slide_line.
    Also returns which rows' results fit in 4 bits per cell. Rows whose slide
    makes a 16 are left to the last section of this file."""
    rows, points, rewards, moved, fits = [], [], [], [], []
    for index in range(N_ROWS):
        row = row_at(index)
        line = row if direction == LEFT else row[::-1]
        new_line, line_points, line_reward = reference_slide_line(line)
        if direction == RIGHT:
            new_line = new_line[::-1]
        rows.append(new_line)
        points.append(line_points)
        rewards.append(line_reward)
        moved.append(new_line != row)
        fits.append(max(new_line) <= 15)
    expected = {
        "rows": np.array(rows),
        "points": np.array(points),
        "rewards": np.array(rewards),
        "moved": np.array(moved),
    }
    return expected, np.array(fits)


@pytest.mark.parametrize("field", TABLE_FIELDS)
@pytest.mark.parametrize("direction", [LEFT, RIGHT], ids=ACTION_NAMES.get)
def test_table_matches_the_rules_for_every_row(direction, field):
    # All 65,536 rows. Also catches a points table whose dtype is too small:
    # [14, 14, 14, 14] is worth 65,536 points.
    expected, fits = expected_table(direction)
    want = expected[field]
    got = np.asarray(table(direction)[TABLE_FIELDS.index(field)])
    assert got.shape == want.shape, f"{field} table has shape {got.shape}, expected {want.shape}"
    wrong = (got != want).reshape(N_ROWS, -1).any(axis=1) & fits
    bad = np.flatnonzero(wrong)
    if len(bad):
        i = bad[0]
        pytest.fail(
            f"{len(bad)} wrong entries in the {ACTION_NAMES[direction]} {field} table, "
            f"e.g. row {row_at(i)} (index {i}): the table has {got[i]}, the rules give {want[i]}"
        )


def test_table_example_row_529():
    # The worked example: [1, 1, 2, 0] is index 529.
    rows, points, rewards, moved = table(LEFT)
    assert list(rows[529]) == [2, 2, 0, 0]
    assert (points[529], rewards[529], moved[529]) == (4, 2, True)
    rows, points, rewards, moved = table(RIGHT)
    assert list(rows[529]) == [0, 0, 2, 2]
    assert (points[529], rewards[529], moved[529]) == (4, 2, True)


def test_playing_doesnt_write_into_the_tables():
    # The tables are shared by every env. If a board is ever a view into a
    # table rather than a copy, spawning a tile writes into the table.
    tables = [t for direction in (LEFT, RIGHT) for t in table(direction)]
    before = [t.copy() for t in tables]
    env = CornerCaseEnv(rng=make_rng(11))
    pick = np.random.default_rng(12)
    for _ in range(500):
        env.step(int(pick.integers(4)))
        if env.game_over():
            env.reset()
    for name, t, b in zip(TABLE_FIELDS * 2, tables, before):
        assert np.array_equal(t, b), f"a {name} table changed during play"


# More lines for the lookup tables, through step() like LINE_CASES: the top of
# the 4-bit range, and big tiles in the last cell (the one weighted 4096 when
# encoding). Not in docs/SPEC.md yet.
TABLE_EDGE_LINE_CASES = [
    ([14, 14, 0, 0], [15, 0, 0, 0], 32768, 15),        # the largest merge a 4-bit cell holds
    ([14, 14, 14, 14], [15, 15, 0, 0], 65536, 30),     # too many points for a 16-bit integer
    ([0, 0, 0, 15], [15, 0, 0, 0], 0, 0),              # a 15 in the last cell
    ([1, 0, 0, 15], [1, 15, 0, 0], 0, 0),
    ([13, 13, 15, 0], [14, 15, 0, 0], 16384, 14),
]

TABLE_EDGE_ILLEGAL_LINES = [
    [15, 0, 0, 0],
    [15, 14, 15, 14],
    [1, 15, 0, 0],
]


@pytest.mark.parametrize("action", ACTIONS, ids=ACTION_NAMES.get)
@pytest.mark.parametrize(
    "line, expected, points, reward", TABLE_EDGE_LINE_CASES,
    ids=[str(c[0]) for c in TABLE_EDGE_LINE_CASES],
)
def test_edge_lines_slide_and_merge(line, expected, points, reward, action):
    env = make_env(place_line(line, action))
    _, got_reward, _, _ = env.step(action)
    assert_slid_then_spawned(env.board, place_line(expected, action))
    assert env.score == points
    assert got_reward == reward


@pytest.mark.parametrize("action", ACTIONS, ids=ACTION_NAMES.get)
@pytest.mark.parametrize("line", TABLE_EDGE_ILLEGAL_LINES, ids=str)
def test_edge_illegal_move_changes_nothing(line, action):
    env = make_env(place_line(line, action))
    before = env.board.copy()
    _, reward, _, _ = env.step(action)
    assert np.array_equal(env.board, before), "an illegal move changed the board or spawned a tile"
    assert env.score == 0
    assert reward == 0
    assert not env.legal_action_mask()[action]


# ---------------------------------------------------------------- tiles above 2^15
#
# A cell gets 4 bits in a table index, so the largest exponent it can hold is
# 15 (a 32768 tile). Without a check, a 16 doesn't fail cleanly. In the last
# cell of a line it pushes the index past the end of the table (an
# IndexError). Anywhere else it carries into the next cell's digit, and the
# engine looks up a different row with no error at all.
#
# The policy, from docs/SPEC.md (Board): a tile above 32768 is refused with a
# ValueError before anything changes. If you move to exact rules later
# (5-bit cells or a slow fallback), replace these with tests that those
# moves come out right.

# The refusal has to name the tile. Otherwise an unrelated ValueError, such as
# NumPy's "truth value of an array is ambiguous", would pass for a refusal.
TILE_TOO_BIG = r"32,?768|65,?536"

BOARDS_ABOVE_32768 = {
    "16 in a corner": [
        [16, 0, 0, 0],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
    ],
    # Full with no merges. An interior cell is never the last cell of a line,
    # so without the check this board is silently wrong, not an IndexError.
    "16 in the interior": [
        [1, 2, 3, 4],
        [2, 16, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ],
    # Above 16 too, so the check has to be "> 15" and not "== 16".
    "17": [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 17, 6],
        [4, 5, 6, 7],
    ],
}


@pytest.mark.parametrize("board", BOARDS_ABOVE_32768.values(), ids=list(BOARDS_ABOVE_32768))
def test_set_board_refuses_a_tile_above_32768(board):
    env = make_env(FULL_BOARD)
    with pytest.raises(ValueError, match=TILE_TOO_BIG):
        env.set_board(np.array(board, dtype=env.board.dtype))
    # The env keeps the board it had, and its cache still matches it.
    assert np.array_equal(env.board, FULL_BOARD)
    assert list(env.legal_action_mask()) == reference_mask(FULL_BOARD)
    step_and_check(env, LEFT)


TWO_32768S = [
    [15, 15, 1, 2],
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
]


@pytest.mark.parametrize("action", [LEFT, RIGHT], ids=ACTION_NAMES.get)
def test_merging_two_32768s_is_refused_without_changing_anything(action):
    # LEFT and RIGHT both merge the 15s in the top row into a 16.
    env = make_env(TWO_32768S)
    env.score = 1000
    with pytest.raises(ValueError, match=TILE_TOO_BIG):
        env.step(action)
    assert np.array_equal(env.board, TWO_32768S), "the board changed before the refusal"
    assert env.score == 1000, "the score changed before the refusal"
    assert list(env.legal_action_mask()) == reference_mask(TWO_32768S)


def test_only_the_chosen_move_is_refused():
    # LEFT would merge the two 32768s, but DOWN merges nothing. A check over
    # all four cached moves (say, while building the cache) would refuse DOWN
    # too, and end a game that can go on.
    env = make_env([
        [15, 15, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    step_and_check(env, DOWN)
