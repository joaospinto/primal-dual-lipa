# LIPA solver comparison

## cartpole

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| csqp-jax |     113 |       22325 |         66.8 |   4.17e-07 |                   0 |          0 |  1.78e-15 |  4.17e-07 |           0 |          0 |   6.72e-10 |       32.3 |  32.3 | ok   |         |


## acrobot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| csqp-jax |      74 |      7516.5 |        44.51 |   4.88e-15 |                   0 |          0 |  4.88e-15 |         0 |           0 |          0 |          0 |       13.6 |  13.6 | ok   |         |


## quadpendulum

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| csqp-jax |    1000 | 1.07741e+06 |        8.694 |   1.87e-06 |            7.84e-05 |          0 |  1.87e-06 |  6.99e-09 |    7.84e-05 |   2.67e-23 |   0.000183 |       6.85 |  6.85 | x    |         |


## quadpendulum_theta

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                           |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------------------------------------------------------|
| csqp-jax |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | CSQP/Crocoddyl does not support cross-stage Theta (theta_dim=1) |


## barrel_roll

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |    KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|--------|------|---------|
| csqp-jax |     200 |      352011 |        22230 |      0.198 |                26.9 |          0 |     0.198 |         0 |        26.9 |   5.55e-17 |       23.1 |     277000 | 277000 | x    |         |


## backflip

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| csqp-jax |     200 |      656878 |        31460 |     0.0126 |                80.6 |          0 |    0.0126 |         0 |        80.6 |   8.88e-17 |       2.15 |      24300 | 24300 | x    |         |


## jump

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------|
| csqp-jax |     400 |      948528 |        19500 |      0.653 |                 103 |          0 |     0.653 |         0 |         103 |          0 |        1.6 |      29200 | 29200 | x    | two-phase warm start |


## trot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |    KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|--------|------|----------------------|
| csqp-jax |     212 |      312338 |        30900 |      0.131 |                2.96 |          0 |     0.131 |         0 |        2.96 |          0 |       5.45 |     277000 | 277000 | x    | two-phase warm start |


## Summary: iterations + status

| solver   | acrobot   | backflip   | barrel_roll   | cartpole   | jump   | quadpendulum   | quadpendulum_theta   | trot   |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|--------|
| csqp-jax | 74 ok     | 200 x      | 200 x         | 113 ok     | 400 x  | 1000 x         | N/A                  | 212 x  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver   | acrobot    | backflip    | barrel_roll   | cartpole    | jump        | quadpendulum   | quadpendulum_theta   | trot        |
|----------|------------|-------------|---------------|-------------|-------------|----------------|----------------------|-------------|
| csqp-jax | 7517 ms ok | 656878 ms x | 352011 ms x   | 22325 ms ok | 948528 ms x | 1077415 ms x   | -                    | 312337 ms x |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver   |   acrobot |   backflip |   barrel_roll |   cartpole |   jump |   quadpendulum | quadpendulum_theta   |   trot |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|--------|
| csqp-jax |      13.6 |      24300 |        277000 |       32.3 |  29200 |           6.85 | -                    | 277000 |

