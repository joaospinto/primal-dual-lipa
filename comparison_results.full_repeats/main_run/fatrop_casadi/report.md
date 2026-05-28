# LIPA solver comparison

## cartpole

| solver        |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   |   notes |
|---------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| fatrop-casadi |      80 |        41.4 |         66.8 |   1.94e-12 |                   0 |          0 |  1.94e-12 |  9.83e-24 |           0 |          0 |   1.78e-07 |   2.83e-11 | 1.78e-07 | ok   |       0 |


## acrobot

| solver        |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   |   kkt:stat |      KKT | ok   |   notes |
|---------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| fatrop-casadi |      17 |        30.3 |        44.51 |   3.16e-11 |                   0 |          0 |  3.16e-11 |         0 |           0 | -          | -          |   3.19e-11 | 3.19e-11 | ok   |       0 |


## quadpendulum

| solver        |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   |   notes |
|---------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| fatrop-casadi |     112 |      1595.5 |        8.279 |   8.38e-12 |                   0 |          0 |  8.38e-12 |  6.21e-21 |           0 |          0 |      1e-07 |   3.45e-11 | 1e-07 | ok   |       0 |


## quadpendulum_theta

| solver        |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                            |
|---------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|------------------------------------------------------------------|
| fatrop-casadi |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | fatrop does not natively support cross-stage Theta (theta_dim=1) |


## Summary: iterations + status

| solver        | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|---------------|-----------|------------|----------------|----------------------|
| fatrop-casadi | 17 ok     | 80 ok      | 112 ok         | N/A                  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver        | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|---------------|-----------|------------|----------------|----------------------|
| fatrop-casadi | 30 ms ok  | 41 ms ok   | 1596 ms ok     | -                    |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver        |   acrobot |   cartpole |   quadpendulum | quadpendulum_theta   |
|---------------|-----------|------------|----------------|----------------------|
| fatrop-casadi |  3.19e-11 |   1.78e-07 |          1e-07 | -                    |

