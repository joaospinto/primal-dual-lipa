# LIPA solver comparison

## cartpole

| solver      |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|-------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| csqp-casadi |     115 |      3929.1 |         66.8 |   4.17e-07 |                   0 |          0 |  8.88e-16 |  4.17e-07 |           0 |          0 |   6.72e-10 |       32.3 |  32.3 | ok   |         |


## acrobot

| solver      |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|-------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| csqp-casadi |      74 |       569.5 |        44.51 |   5.33e-15 |                   0 |          0 |  5.33e-15 |         0 |           0 |          0 |          0 |       13.6 |  13.6 | ok   |         |


## quadpendulum

| solver      |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|-------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| csqp-casadi |    1000 |      315738 |        8.694 |   1.82e-06 |            8.03e-05 |          0 |  1.82e-06 |  7.08e-09 |    8.03e-05 |          0 |   0.000188 |       6.86 |  6.86 | x    |         |


## quadpendulum_theta

| solver      |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                           |
|-------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------------------------------------------------------|
| csqp-casadi |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | CSQP/Crocoddyl does not support cross-stage Theta (theta_dim=1) |


## Summary: iterations + status

| solver      | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|-------------|-----------|------------|----------------|----------------------|
| csqp-casadi | 74 ok     | 115 ok     | 1000 x         | N/A                  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver      | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|-------------|-----------|------------|----------------|----------------------|
| csqp-casadi | 570 ms ok | 3929 ms ok | 315738 ms x    | -                    |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver      |   acrobot |   cartpole |   quadpendulum | quadpendulum_theta   |
|-------------|-----------|------------|----------------|----------------------|
| csqp-casadi |      13.6 |       32.3 |           6.86 | -                    |

