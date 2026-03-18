# Primal-Dual Lagrangian Interior Point Algorithm (LIPA)

## Running the unit tests

The unit tests can be run with `uv run python -m unittest`.
This handles the environment set up properly and runs all tests.

It's important to check the solver logs in the unit tests, and not look only
at the exit status. This is especially true if you modify passing criteria (which you should avoid).

If `uv` is not installed, install it via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
Make sure to request the user for permission to install it.
If this doesn't work, refer to the `Installation` section in https://github.com/astral-sh/uv.

## Mathematical theory behind this library

The mathematical derivations behind this method can be found in
https://arxiv.org/html/2509.16370v5.
This repository, together with https://github.com/joaospinto/regularized_lqr_jax
(which we use here as a dependency), implement all of the contents of that paper.
However, keep in mind that the feature requests may ask you to depart from this.

## Boundaries

- Never push any changes to the git remote.
- Use local feature branches to implement your changes.
- Do not edit the git history prior to the start of your task.

## Development guidelines

- When being asked to perform a task, first create a PLAN.md file in the project directory;
  document the codebase research, formulate a plan, and wait for user approval before implementing any code.
- Make minimal changes to implement the requested features.
- When appropriate, use feature flags in the existing settings data classes.
- Be methodical; numerical optimization is a hard subject, so you should operate based on facts and not vibes.
- Review your own work; when you think you're done, check:
  1. What did you actually change in the code?
  2. Do the code changes actually implement the feature request?
  3. Do the unit tests pass, and do the solver logs look good (not only the test exit status)?
  4. Do the unit tests adequately cover the feature request?
  5. Are there good unit tests covering the feature request in isolation?
  6. Did you cheat by modifying the success criteria or lowering expectations? This is completely unacceptable.
