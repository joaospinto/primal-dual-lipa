# LIPA solver comparison

## cartpole

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   |   notes |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| fatrop-jax |      99 |       37155 |         66.8 |   4.74e-11 |                   0 |          0 |  4.74e-11 |  2.65e-23 |           0 |          0 |   1.06e-07 |   4.31e-10 | 1.06e-07 | ok   |       0 |


## acrobot

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   |   kkt:stat |      KKT | ok   |   notes |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| fatrop-jax |      17 |      4265.3 |        44.51 |   3.16e-11 |                   0 |          0 |  3.16e-11 |         0 |           0 | -          | -          |   3.19e-11 | 3.19e-11 | ok   |       0 |


## quadpendulum

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   |   notes |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| fatrop-jax |     112 |      189991 |        8.279 |   8.38e-12 |                   0 |          0 |  8.38e-12 |  3.43e-21 |           0 |          0 |      1e-07 |   3.45e-11 | 1e-07 | ok   |       0 |


## quadpendulum_theta

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                            |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|------------------------------------------------------------------|
| fatrop-jax |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | fatrop does not natively support cross-stage Theta (theta_dim=1) |


## Summary: iterations + status

| solver     | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|------------|-----------|------------|----------------|----------------------|
| fatrop-jax | 17 ok     | 99 ok      | 112 ok         | N/A                  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver     | acrobot    | cartpole    | quadpendulum   | quadpendulum_theta   |
|------------|------------|-------------|----------------|----------------------|
| fatrop-jax | 4265 ms ok | 37155 ms ok | 189991 ms ok   | -                    |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver     |   acrobot |   cartpole |   quadpendulum | quadpendulum_theta   |
|------------|-----------|------------|----------------|----------------------|
| fatrop-jax |  3.19e-11 |   1.06e-07 |          1e-07 | -                    |

