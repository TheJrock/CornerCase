"""
Measures how good an agent actually is.

Usage (once implemented):  python evaluate.py <checkpoint> OPTIONAL: <num_games>

Responsibilities:
  - Load a checkpoint and play N games greedily (epsilon = 0).
  - Report mean and median score, the distribution of max tile reached (% of
    games reaching 512, 1024, 2048...), and average game length.

Build baselines first and compare against them:
  - Random agent (a random legal move each turn).
  - A simple heuristic, such as a fixed preference order like
    down > left > right > up. That's the "corner strategy" this repo is named for.
If your DQN can't beat the heuristic, something is wrong.
"""

import time
import numpy as np
from agent import RandomAgent, PatternHeuristicAgent, CornerHeuristicAgent
from game import CornerCaseEnv, MAX_EXPONENT

def evaluate(agent, num_games):
    scores = []
    tile_counts = {2 ** i: 0 for i in range(5, MAX_EXPONENT + 1)}
    game_lengths = []

    start_time = time.perf_counter()
    for seed in range(num_games):
        agent_seed, env_seed = np.random.SeedSequence(seed).spawn(2)
        agent_rng = np.random.default_rng(agent_seed)
        env_rng = np.random.default_rng(env_seed)
        agent.reset(rng=agent_rng)
        env = CornerCaseEnv(rng=env_rng)
        state = env.board.copy()
        legal_action_mask = env.legal_action_mask()
        done = False
        game_length = 0
        while not done:
            action = agent.act(legal_action_mask, state)
            if not legal_action_mask[action]:
                raise RuntimeError(f"Agent chose an illegal action: {action}")
            state, _, done, info = env.step(action)
            legal_action_mask = info["legal_action_mask"]
            game_length += 1
        scores.append(env.score)
        largest_tile = int(2 ** env.board.max())
        for tile in tile_counts:
            if largest_tile >= tile:
                tile_counts[tile] = tile_counts[tile] + 1
        game_lengths.append(game_length)

    end_time = time.perf_counter()
    moves_per_second = sum(game_lengths) / (end_time - start_time)

    tile_rates = {tile: count / num_games for tile, count in tile_counts.items()}

    return np.mean(scores), np.median(scores), tile_rates, np.mean(game_lengths), moves_per_second

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python evaluate.py <checkpoint> OPTIONAL: <num_games>")
        sys.exit(1)
    checkpoint = sys.argv[1]
    num_games = int(sys.argv[2]) if len(sys.argv) == 3 else 100
    agents = []
    match checkpoint.lower():
        case "all":
            agents = [RandomAgent(), PatternHeuristicAgent(), CornerHeuristicAgent()]
        case "random":
            agents = [RandomAgent()]
        case "pattern":
            agents = [PatternHeuristicAgent()]
        case "corner":
            agents = [CornerHeuristicAgent()]
        case _:
            raise ValueError(f"Unknown checkpoint: {checkpoint}")

    for agent in agents:
        mean_score, median_score, tile_rates, mean_game_length, moves_per_second = evaluate(agent, num_games)
        print("Agent:", type(agent).__name__)
        print("Mean Score:", mean_score)
        print("Median Score:", median_score)
        print("Tile Rates:", tile_rates)
        print("Mean Game Length:", mean_game_length)
        print("Moves per Second:", moves_per_second)