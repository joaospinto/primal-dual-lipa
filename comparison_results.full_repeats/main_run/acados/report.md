# LIPA solver comparison

## cartpole

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   | kkt:stat   |      KKT | ok   | notes    |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|----------|
| acados   |      68 |        57.3 |         66.8 |   5.33e-15 |                   0 |          0 |  5.33e-15 |  4.44e-16 |           0 | -          | -          | -          | 5.33e-15 | ok   | status=0 |


## acrobot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   |   kkt:stat |   KKT | ok   | notes    |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------|
| acados   |      99 |        11.8 |        44.51 |   3.68e-13 |                   0 |          0 |  3.68e-13 |         0 |           0 | -          | -          |       13.6 |  13.6 | ok   | status=0 |


## quadpendulum

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   | kkt:stat   |     KKT | ok   | notes    |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|---------|------|----------|
| acados   |      31 |         527 |        11.75 |    1.2e-11 |                   0 |          0 |   1.2e-11 |  1.39e-12 |           0 | -          | -          | -          | 1.2e-11 | ok   | status=0 |


## quadpendulum_theta

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                            |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|------------------------------------------------------------------|
| acados   |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | acados does not natively support cross-stage Theta (theta_dim=1) |


## Summary: iterations + status

| solver   | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|----------|-----------|------------|----------------|----------------------|
| acados   | 99 ok     | 68 ok      | 31 ok          | N/A                  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver   | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|----------|-----------|------------|----------------|----------------------|
| acados   | 12 ms ok  | 57 ms ok   | 527 ms ok      | -                    |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver   |   acrobot |   cartpole |   quadpendulum | quadpendulum_theta   |
|----------|-----------|------------|----------------|----------------------|
| acados   |      13.6 |   5.33e-15 |        1.2e-11 | -                    |

