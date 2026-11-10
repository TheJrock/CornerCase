# CornerCase
A Deep RL Agent for 2048

A DQN agent trained in Python to play 2048, with a static web demo where you can watch it play in the browser.

## Layout

```
training/   Python: game engine, DQN, training loop, evaluation, export
web/        Static site: game, rendering, in-browser inference
docs/       SPEC.md, the contract both sides must agree on
.github/    Deploy workflow (web/ -> GitHub Pages)
```

Training happens offline. The only thing that crosses from `training/` to `web/` is the exported model in `web/models/`.

Build plan and progress: [docs/ROADMAP.md](docs/ROADMAP.md)
