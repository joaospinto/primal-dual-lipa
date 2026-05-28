# LIPA solver comparison

## cartpole

| solver          |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes   |
|-----------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| aligator-casadi |     101 |      1299.5 |         66.8 |   1.16e-09 |            4.26e-14 | -          | -         | -         | -           | -          | -          | -          | -     | ok   |         |


## acrobot

| solver          |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes   |
|-----------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| aligator-casadi |      55 |       514.8 |        44.51 |   4.24e-08 |                   0 | -          | -         | -         | -           | -          | -          | -          | -     | ok   |         |


## quadpendulum

| solver          |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes   |
|-----------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| aligator-casadi |    1000 |      154943 |        22.45 |     0.0199 |              0.0398 | -          | -         | -         | -           | -          | -          | -          | -     | x    |         |


## quadpendulum_theta

| solver          |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                     |
|-----------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------------------------------------------------|
| aligator-casadi |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | aligator does not support cross-stage Theta (theta_dim=1) |


## Summary: iterations + status

| solver          | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|-----------------|-----------|------------|----------------|----------------------|
| aligator-casadi | 55 ok     | 101 ok     | 1000 x         | N/A                  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver          | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|-----------------|-----------|------------|----------------|----------------------|
| aligator-casadi | 515 ms ok | 1299 ms ok | 154943 ms x    | -                    |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver          | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|-----------------|-----------|------------|----------------|----------------------|
| aligator-casadi | -         | -          | -              | -                    |

