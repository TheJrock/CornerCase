"""
Converts a trained checkpoint into a file the browser can load.

Usage (once implemented):  python export.py <checkpoint>

Writes one JSON file per checkpoint to web/models/<name>.json. Those files are
the only training output that ships to the site.

Format (decided, see docs/SPEC.md -> Model file for the exact structure):
  - meta: name, training moves so far, evaluation mean score. The site uses it
    to label checkpoints for the "watch it learn" feature.
  - layers, input to output: each weight and bias as a shape plus a flat
    "data" array. Write each weight the way PyTorch stores nn.Linear.weight,
    [out, in], row-major.
  - fixtures: a few boards with the model's raw Q-values (before masking).
  - Numbers with 9 significant digits. tolist() writes 17-digit doubles,
    which nearly doubles the download; fewer than 9 digits makes the
    browser's Q-values drift from PyTorch's.

If the model has a dueling head, fold it into one [4, H] Linear here (formula
in SPEC.md -> Network), so the JS side only ever sees a plain MLP.

Gotcha: refuse to export if any weight is NaN or infinite. Python's json
module writes them as NaN / Infinity, which JSON.parse rejects, so the site
would fail to load the model rather than this script failing.

Verify the export: load the file you just wrote, run the fixture boards
through it with plain numpy, and confirm the Q-values match the PyTorch
model's. Pick fixture boards with empty cells and large tiles, so they cover
the whole first-layer lookup.

Size: each checkpoint at 256 wide is about 0.7 MB as sent (GitHub Pages
compresses JSON in transit). The browser fetches only the checkpoints the page
actually loads.
"""
