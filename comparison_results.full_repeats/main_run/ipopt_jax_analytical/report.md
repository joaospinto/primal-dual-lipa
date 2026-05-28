# LIPA solver comparison

## cartpole

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes           |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------|
| ipopt-jax |      46 |      5638.5 |         66.8 |    3.4e-11 |                   0 |   1.88e-30 |   3.4e-11 |  2.85e-35 |           0 |       1.27 |       35.9 |       6.68 |  35.9 | ok   | Solve_Succeeded |


## acrobot

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   |   kkt:stat |      KKT | ok   | notes           |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|-----------------|
| ipopt-jax |      21 |      1924.5 |        44.51 |   5.48e-07 |                   0 |   2.84e-29 |  5.48e-07 |         0 |           0 | -          | -          |   5.35e-07 | 5.48e-07 | ok   | Solve_Succeeded |


## quadpendulum

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes           |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------|
| ipopt-jax |     163 |     95391.4 |        11.75 |   2.07e-08 |                   0 |   5.86e-24 |  2.07e-08 |  1.65e-27 |           0 |      0.243 |       14.9 |       9.37 |  14.9 | ok   | Solve_Succeeded |


## quadpendulum_theta

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes           |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------|
| ipopt-jax |      82 |     51383.3 |        14.81 |   2.62e-07 |            8.24e-09 |   2.77e-28 |  2.62e-07 |  4.45e-31 |    8.24e-09 |      0.169 |        240 |        174 |   240 | ok   | Solve_Succeeded |


## Summary: iterations + status

| solver    | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|-----------|-----------|------------|----------------|----------------------|
| ipopt-jax | 21 ok     | 46 ok      | 163 ok         | 82 ok                |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver    | acrobot    | cartpole   | quadpendulum   | quadpendulum_theta   |
|-----------|------------|------------|----------------|----------------------|
| ipopt-jax | 1924 ms ok | 5638 ms ok | 95391 ms ok    | 51383 ms ok          |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver    |   acrobot |   cartpole |   quadpendulum |   quadpendulum_theta |
|-----------|-----------|------------|----------------|----------------------|
| ipopt-jax |  5.48e-07 |       35.9 |           14.9 |                  240 |

