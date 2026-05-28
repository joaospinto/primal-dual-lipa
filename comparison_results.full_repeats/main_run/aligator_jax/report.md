# LIPA solver comparison

## cartpole

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes   |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| aligator-jax |     114 |     16403.8 |         66.8 |   1.16e-09 |            4.26e-14 | -          | -         | -         | -           | -          | -          | -          | -     | ok   |         |


## acrobot

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes   |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| aligator-jax |      55 |      4988.9 |        44.51 |   4.24e-08 |                   0 | -          | -         | -         | -           | -          | -          | -          | -     | ok   |         |


## quadpendulum

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                       |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------------------------------------------|
| aligator-jax |       0 |      360529 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 360s (subprocess timeout) |


## quadpendulum_theta

| solver       |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                     |
|--------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------------------------------------------------|
| aligator-jax |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | aligator does not support cross-stage Theta (theta_dim=1) |


## Summary: iterations + status

| solver       | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|--------------|-----------|------------|----------------|----------------------|
| aligator-jax | 55 ok     | 114 ok     | 0 x            | N/A                  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver       | acrobot    | cartpole    | quadpendulum   | quadpendulum_theta   |
|--------------|------------|-------------|----------------|----------------------|
| aligator-jax | 4989 ms ok | 16404 ms ok | -              | -                    |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver       | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|--------------|-----------|------------|----------------|----------------------|
| aligator-jax | -         | -          | -              | -                    |

