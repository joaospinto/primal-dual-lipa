# Primal-Dual Lagrangian Interior Point Method (LIPA)

<p align="center">
  <img alt="Primal-Dual LIPA Logo" src="https://github.com/user-attachments/assets/ed197b8f-08ae-41f9-8433-470bff46284c" width="50%" />
</p>

<p align="center">
  <img alt="Quadpendulum" src="quadpend_timelapse.png" width="45%" />
  <img alt="Cartpole" src="cartpole_timelapse.png" width="45%" />
</p>

This repository implements a numerical method for solving
discrete-time optimal control problems in [JAX](https://github.com/jax-ml/jax),
leveraging the techniques developed in
[Dual-Regularized Riccati Recursions for Interior-Point Optimal Control](https://arxiv.org/abs/2509.16370v5),
and putting to use the [Regularized LQR in JAX](https://github.com/joaospinto/regularized_lqr_jax) code.

This library supports arbitrary non-convex costs, constraints, and dynamics.
The costs and constraints are expected to be stage-wise, but the solver
efficiently supports "cross-stage" optimization variables.

We draw inspiration from [Trajax](https://github.com/google/trajax)
and [MPX](https://github.com/iit-DLSLab/mpx/tree/main).
Here are some ways in which we improve on these libraries:

| Feature                         | Trajax   | MPX     | LIPA |
| ------------------------------- | -------- | ------- | ---- |
| Multiple shooting               | ❌       | ✅      | ✅   |
| Efficient GPU usage             | ❌       | ✅      | ✅   |
| Parallel line search            | ❌       | ✅      | ✅   |
| IPM for inequalities            | ❌       | ❌      | ✅   |
| Cross-stage terms               | ❌       | ❌      | ✅   |
| Iterative refinement            | ❌       | ❌      | ✅   |
| Per-constraint $\rho$           | ❌       | ❌      | ✅   |
| Consistent $\rho$ in LS and KKT | ✅       | ❌      | ✅   |

## Installation

If you just want to try out the examples in this repository, we suggest
that you [install uv](https://docs.astral.sh/uv/getting-started/installation/).

If you want to add a project dependency on this repository (or `pip install` it), you can use:

```"primal-dual-lipa @ git+https://github.com/joaospinto/primal-dual-lipa"```

## Examples

The examples below are based on the [Trajax](https://github.com/google/trajax)
tests and notebooks, but are modified to make use of some of the features
that are enabled by our solver.

### Quadpendulum

This example, which can be run with `uv run --extra test python -m unittest tests/test_quadpendulum.py`,
optimizes the trajectory of a quadpendulum to reach a target goal-state,
while maximizing the worst-case distance to some circular obstacles.
This requires using cross-stage variables in both the costs and in the constraints.

https://github.com/user-attachments/assets/bba4ad30-25d7-434e-b8cb-861a9fddf0d8

### Backflip

This example, which can be run with
`uv run --extra mpc-examples python -m tests.mpc_examples.run_offline --task h1_backflip --video h1_backflip.mp4`,
optimizes the trajectory of a humanoid robot doing a backflip.

https://github.com/user-attachments/assets/1184ad08-3d06-465b-aca5-5db6e8cb5b0f

### Barrel Roll

This example, which can be run with
`uv run --extra mpc-examples python -m tests.mpc_examples.run_offline --task barrel_roll --video barrel_roll.mp4`,
optimizes the trajectory of a four-legged robot doing a barrel roll.

https://github.com/user-attachments/assets/6b7ffda4-0a14-42ae-8234-8c882b0fb991

### Trotting

This example, which can be run with
`uv run --extra mpc-examples python -m tests.mpc_examples.run_offline --task aliengo_trot --video aliengo_trot.mp4`,
optimizes the trajectory of a four-legged robot trotting.

https://github.com/user-attachments/assets/07b234f9-cd09-4090-a4b0-9ba63a72cd13

### Humanoid jump

This example, which can be run with
`uv run --extra mpc-examples python -m tests.mpc_examples.run_offline --task h1_jump_forward --video h1_jump_forward.mp4`,
optimizes the trajectory of a humanoid doing two consecutive jumps.

https://github.com/user-attachments/assets/78c329e5-c233-4364-9041-3b3fd0689671

### Cartpole

This example, which can be run with `uv run --extra test python -m unittest tests/test_cartpole.py`,
optimizes the trajectory of a quadpendulum to reach a target goal-state,
while imposing some control bounds.

https://github.com/user-attachments/assets/25efb120-253e-48b0-ac14-1dd38f385334

## Citing this work

```bibtex
@misc{2025dualregularizedlqr,
      title={Dual-Regularized Riccati Recursions for Interior-Point Optimal Control}, 
      author={João Sousa-Pinto and Dominique Orban},
      year={2025},
      eprint={2509.16370},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2509.16370},
}
```
