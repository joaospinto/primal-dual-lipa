# LIPA solver comparison

## cartpole

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes           |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|-----------------|
| ipopt-casadi |      33 |        58.2 |         66.8 |   2.51e-10 |                   0 |   3.41e-34 |  2.51e-10 |  3.74e-37 |           0 |          0 |   9.82e-08 |   9.15e-09 | 9.82e-08 | ok   | Solve_Succeeded |


## acrobot

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes           |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|-----------------|
| ipopt-casadi |      21 |        33.5 |        44.51 |   5.48e-07 |                   0 |   6.31e-30 |  5.48e-07 |         0 |           0 |          0 |          0 |   5.34e-07 | 5.48e-07 | ok   | Solve_Succeeded |


## quadpendulum

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes           |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|-----------------|
| ipopt-casadi |      65 |      4000.4 |        8.279 |   3.48e-12 |               1e-08 |   4.95e-31 |  3.48e-12 |  1.37e-32 |       1e-08 |          0 |   1.97e-08 |   2.21e-11 | 1.97e-08 | ok   | Solve_Succeeded |


## quadpendulum_theta

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes           |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|-----------------|
| ipopt-casadi |     103 |      6768.9 |        6.907 |   1.68e-09 |                   0 |   4.58e-26 |  1.68e-09 |  2.31e-31 |           0 |          0 |   1.04e-07 |   1.06e-07 | 1.06e-07 | ok   | Solve_Succeeded |


## Summary: iterations + status

| solver       | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|--------------|-----------|------------|----------------|----------------------|
| ipopt-casadi | 21 ok     | 33 ok      | 65 ok          | 103 ok               |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver       | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|--------------|-----------|------------|----------------|----------------------|
| ipopt-casadi | 34 ms ok  | 58 ms ok   | 4000 ms ok     | 6769 ms ok           |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver       |   acrobot |   cartpole |   quadpendulum |   quadpendulum_theta |
|--------------|-----------|------------|----------------|----------------------|
| ipopt-casadi |  5.48e-07 |   9.82e-08 |       1.97e-08 |             1.06e-07 |

