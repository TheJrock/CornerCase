/*
  Entry point that connects everything. Loaded by index.html.

  Responsibilities:
    - Start playing on page load, with no clicks needed: create a game on a
      random seed and draw it right away, while the model file loads in the
      background (the latest checkpoint, if you ship several). Show a small
      loading indicator until the model is ready, then the agent starts
      playing on its own.
    - The play loop: on a timer, ask the agent for a move, apply it, render,
      and stop at game over.
    - Hook up the controls: play/pause, step, speed, new game (and keyboard
      input if humans can play).

  Use setTimeout, or requestAnimationFrame with a delay. The agent picks a move
  in microseconds, so you have to slow it down to make it watchable.
*/
