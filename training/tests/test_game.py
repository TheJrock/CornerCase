"""
Tests for the game engine. Write these before (or while) you write game.py.

Run from training/ with:  pytest

Cases worth covering:
  - Row merges when sliding left:
      [2, 2, 2, 2]  -> [4, 4, 0, 0]   pairs merge from the leading edge
      [2, 2, 4, 0]  -> [4, 4, 0, 0]   a newly merged tile can't merge again in the same move
      [4, 0, 0, 4]  -> [8, 0, 0, 0]   tiles slide past gaps before merging
      [2, 4, 8, 16] -> unchanged      not a legal move
  - The same cases in the other three directions.
  - Points gained equal the sum of the newly merged tiles.
  - Exactly one tile spawns after a legal move, and none after an illegal one.
  - Legal-move detection and game-over detection on hand-made boards.
  - The same seed gives the same game.

These cases are also listed in docs/SPEC.md so the JS engine gets checked
against the same expectations.
"""
