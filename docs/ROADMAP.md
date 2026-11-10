# Roadmap

Work through these roughly in order. Each file's header comment has the details for that file, and the decisions go in [SPEC.md](SPEC.md).

## 1. Commit the scaffolding
- [ ] Review the placeholder files and commit them
- [ ] Make sure the repo is public (GitHub Pages on the free plan requires it)

## 2. Set up Python
Use Homebrew's Python 3.12 with a plain venv, not conda. Run from `training/`:
```bash
conda deactivate                          # if (base) is showing
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate                 # `source`, not `python`
which python                              # should end in training/.venv/bin/python
```
- [ ] venv created and activated
- [ ] Dependencies chosen in `requirements.txt`, then `pip install -r requirements.txt`

## 3. Make the first spec decisions
- [ ] Board representation (values or exponents) in SPEC.md
- [ ] Action index order in SPEC.md

## 4. Game engine and tests
`game.py`, `tests/test_game.py`
- [ ] Tests for merge rules, spawning, legal moves, game over
- [ ] `pytest` passes
- [ ] Get it correct first, then make it fast: training runs millions of moves through it

## 5. Baselines
`evaluate.py`
- [ ] Random agent
- [ ] Corner heuristic
- [ ] Record their mean score and max-tile distribution: these are the numbers DQN has to beat
- [ ] Note moves per second, which tells you whether the engine is fast enough to train with

## 6. DQN
`encoding.py` → `network.py` → `replay_buffer.py` → `agent.py` → `config.py` → `train.py`
- [ ] Remaining SPEC decisions: encoding, reward, network shape
- [ ] Short end-to-end run with a tiny config (finds bugs in minutes instead of hours)
- [ ] Full training run, keeping checkpoints from early, mid and late in training
- [ ] Beats the corner heuristic in `evaluate.py`

## 7. Web demo (you can start this while step 6 trains)
- [ ] `export.py` writes the model and Q-value fixtures to `web/models/`
- [ ] `game.js` passes the golden test cases in SPEC.md
- [ ] `renderer.js` + `main.js`: a human can play with the arrow keys
- [ ] `encoding.js` + `agent.js`: Q-values match the fixtures from `export.py`
- [ ] The agent plays in the browser

Test locally with `cd web && python -m http.server 8000`, then open http://localhost:8000. Opening `index.html` directly won't work.

## 8. Deploy to the subdomain
You can do this any time after step 1, but **keep the sub-steps in this order**:
- [ ] Fill in `.github/workflows/deploy.yml.todo`, then rename it to `deploy.yml`
- [ ] Repo Settings → Pages → Source: **GitHub Actions**. Push, then check the site loads at the URL shown there
- [ ] Repo Settings → Pages → Custom domain: `<subdomain>.<yourdomain>`
- [ ] **Only then** add the DNS CNAME record `<subdomain>` → `<username>.github.io`
- [ ] Turn on Enforce HTTPS once the certificate is issued

If you add the DNS record before the repo claims the subdomain, someone else can claim it. Verifying your domain under your GitHub account's Settings → Pages protects every subdomain, and you can do that at any time.
