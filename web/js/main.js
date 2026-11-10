/*
  Entry point that connects everything. Loaded by index.html.

  Responsibilities:
    - Load the agent's model on startup and show a loading state until it's ready.
    - Create a game and render it.
    - The play loop: on a timer, ask the agent for a move, apply it, render,
      and stop at game over.
    - Hook up the controls: play/pause, step, speed, new game (and keyboard
      input if humans can play).

  Use setTimeout, or requestAnimationFrame with a delay. The agent picks a move
  in microseconds, so you have to slow it down to make it watchable.
*/
