# Roadmap

Work through these roughly in order. Each file's header comment has the details for that file, and the decisions go in [SPEC.md](SPEC.md).

## 1. Commit the scaffolding
- [x] Review the placeholder files and commit them
- [x] Make sure the repo is public (GitHub Pages on the free plan requires it)

## 2. Set up Python
Use Homebrew's Python 3.12 with a plain venv, not conda. Run from `training/`:
```bash
conda deactivate                          # if (base) is showing
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate                 # `source`, not `python`
which python                              # should end in training/.venv/bin/python
```
- [x] venv created and activated
- [x] Dependencies chosen in `requirements.txt`, then `pip install -r requirements.txt`

## 3. Make the first spec decisions
- [x] Board representation (values or exponents) in SPEC.md
- [x] Action index order in SPEC.md

## 4. Game engine and tests
`game.py`, `tests/test_game.py`
- [x] Tests for merge rules, spawning, legal moves, game over
- [x] Spawning uses a seeded RNG you pass into the game, not the global `random` (see SPEC.md → Tile spawning)
- [x] `pytest` passes
- [x] Get it correct first, then make it fast: training runs millions of moves through it

## 5. Baselines
`evaluate.py`
- [x] Random agent
- [x] Corner heuristic
- [x] Record their mean score and max-tile distribution: these are the numbers DQN has to beat
- [x] Note moves per second, which tells you whether the engine is fast enough to train with

## 6. DQN
`encoding.py` → `network.py` → `replay_buffer.py` → `agent.py` → `config.py` → `train.py`
- [x] Remaining SPEC decisions: encoding, reward, network shape
- [ ] `encoding.py` only flattens boards to [N, 16]. The one-hot is the first step of the forward pass in `network.py` (see SPEC.md → Network input)
- [ ] Replay buffer stores raw uint8 boards, [capacity, 16] each, for the state, the afterstate (after the slide, before the spawn) and the next state, so step 10 needs no buffer changes. The network does the one-hot on each sampled batch, so the buffer never holds encoded boards: 1M transitions take about 48 MB raw, against about 3 GB encoded
- [ ] Device is a `config.py` setting, and the code runs on either `cpu` or `mps`. Train on CPU: on the M1 Pro with this network, the GPU was 15× slower at choosing a move and slower at batch-512 learning steps (benchmarked 2026-11-11). Benchmark again if the network grows a lot
- [ ] Short end-to-end run with a tiny config (finds bugs in minutes instead of hours)
- [ ] Full training run, keeping checkpoints from early, mid and late in training
- [ ] Beats the corner heuristic in `evaluate.py`

Learning steps limit training speed, not the engine: about 430 per second at batch 512 on CPU, against about 33,000 game moves per second. How many moves you play per learning step (set in `config.py`) mostly decides how long a run takes.

## 7. Web demo (you can start this while step 6 trains)
- [x] Choose the model file format: one JSON file per checkpoint (SPEC.md → Model file)
- [ ] `export.py` writes each checkpoint to one JSON file in `web/models/`, with its metadata and Q-value fixtures, folding a dueling head into one Linear if you added one (SPEC.md → Network)
- [ ] `game.js` passes the golden test cases in SPEC.md
- [ ] `renderer.js` + `main.js`: a human can play with the arrow keys
- [ ] `encoding.js` flattens the board. `agent.js` is a hand-written forward pass that computes the first layer as a lookup (SPEC.md → Model file). Q-values match the fixtures inside the model file
- [ ] The agent plays in the browser: on page load a game starts on a random seed, and the agent plays on its own once the model has loaded. Visitors don't have to click anything

Test locally with `cd web && python -m http.server 8000`, then open http://localhost:8000. Opening `index.html` directly won't work.

## 8. Deploy to the subdomain
You can do this any time after step 1, but **keep the sub-steps in this order**:
- [ ] Fill in `.github/workflows/deploy.yml.todo`, then rename it to `deploy.yml`
- [ ] Repo Settings → Pages → Source: **GitHub Actions**. Push, then check the site loads at the URL shown there
- [ ] Repo Settings → Pages → Custom domain: `<subdomain>.<yourdomain>`
- [ ] **Only then** add the DNS CNAME record `<subdomain>` → `<username>.github.io`
- [ ] Turn on Enforce HTTPS once the certificate is issued

If you add the DNS record before the repo claims the subdomain, someone else can claim it. Verifying your domain under your GitHub account's Settings → Pages protects every subdomain, and you can do that at any time.

## 9. N-tuple benchmark (after DQN beats the corner heuristic)
A second, non-neural agent to compare DQN against, side by side in the browser.
- [ ] Afterstate TD(0) learner: a few tuples, each shared across the 8 board symmetries
- [ ] Tuple sizes small enough to ship to the browser (a 4-tuple table is 256 KB, a 6-tuple table is 64 MB)
- [ ] Reuse `game.py`'s packed row index (4 bits per cell, `board @ WEIGHTS`) as the table index: a row or column is already a 4-tuple index
- [ ] Engine fast enough for ~10⁸ training moves (numba or a bitboard engine)
- [ ] DQN vs. n-tuple in `evaluate.py` on the same set of seeds
- [ ] Browser: both agents play side by side from the same seed

## 10. Experiment: spawn noise in the TD target (any time after step 6)
Does removing the randomness of the spawn from the target speed up learning? Everything except the target stays the same.
- [ ] A `target_mode` setting in `config.py`:
  - `sampled`: standard DQN, bootstrapping from the real next state
  - `expected`: probability-weighted average over every possible spawn on the afterstate
  - `afterstate` (optional control): bootstrap from the afterstate with no spawn, which should overestimate Q
- [ ] `game.py` can list every spawn outcome of a board with its probability, tested: 2 × empties outcomes, probabilities sum to 1
- [ ] 3 to 5 training seeds per mode, compared at equal environment steps, with wall-clock time noted
- [ ] Periodic greedy evaluation on one fixed set of evaluation seeds for every run
- [ ] Logged per run: evaluation score and max tile, TD loss, TD-error spread, and predicted Q against the actual return
- [ ] A short write-up of what you found
