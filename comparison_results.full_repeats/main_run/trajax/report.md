# LIPA solver comparison

## cartpole

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                                             |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------------------------------------------------|
| trajax   |     310 |      5396.1 |         66.8 |   1.33e-07 |                   0 |          0 |  8.88e-16 |  1.33e-07 |           0 |          0 |   3.67e-14 |       5.57 |  5.57 | ok   | al_iters=8 ilqr_iters_total=310 max_viol=1.33e-07 |


## acrobot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   |   kkt:stat |   KKT | ok   | notes                            |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------|
| trajax   |      86 |       876.4 |        44.51 |   4.44e-16 |                   0 |          0 |  4.44e-16 |         0 |           0 | -          | -          |       6.78 |  6.78 | ok   | ilqr_iters=86 grad_norm=8.61e-07 |


## quadpendulum

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                                             |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------------------------------------------------|
| trajax   |     237 |     12248.3 |        9.354 |   1.78e-08 |            6.15e-07 |          0 |  8.88e-16 |  1.78e-08 |    6.15e-07 |          0 |   7.26e-07 |       2.52 |  2.52 | ok   | al_iters=5 ilqr_iters_total=237 max_viol=6.15e-07 |


## quadpendulum_theta

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                                            |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|------------------------------------------------------------------|
| trajax   |       0 |           0 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | trajax does not natively support cross-stage Theta (theta_dim=1) |


## barrel_roll

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                        |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------------------|
| trajax   |       0 | 2.40051e+06 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 2400s (subprocess timeout) |


## backflip

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                                             |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------------------------------------------------|
| trajax   |     196 |      240760 |        41740 |   8.88e-16 |            2.08e-05 |          0 |  8.88e-16 |         0 |    2.08e-05 |          0 |   0.000566 |       3200 |  3200 | ok   | al_iters=2 ilqr_iters_total=196 max_viol=2.08e-05 |


## jump

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                                                                   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-------------------------------------------------------------------------|
| trajax   |    1027 | 1.04363e+06 |        19640 |   1.68e-15 |            0.000374 |          0 |  1.68e-15 |         0 |    0.000374 |          0 |    0.00014 |       3200 |  3200 | ok   | two-phase warm start; al_iters=1 ilqr_iters_total=977 max_viol=3.74e-04 |


## trot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                        |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------------------|
| trajax   |       0 |  2.4006e+06 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 2400s (subprocess timeout) |


## Summary: iterations + status

| solver   | acrobot   | backflip   | barrel_roll   | cartpole   | jump    | quadpendulum   | quadpendulum_theta   | trot   |
|----------|-----------|------------|---------------|------------|---------|----------------|----------------------|--------|
| trajax   | 86 ok     | 196 ok     | 0 x           | 310 ok     | 1027 ok | 237 ok         | N/A                  | 0 x    |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver   | acrobot   | backflip     | barrel_roll   | cartpole   | jump          | quadpendulum   | quadpendulum_theta   | trot   |
|----------|-----------|--------------|---------------|------------|---------------|----------------|----------------------|--------|
| trajax   | 876 ms ok | 240760 ms ok | -             | 5396 ms ok | 1043632 ms ok | 12248 ms ok    | -                    | -      |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver   |   acrobot |   backflip | barrel_roll   |   cartpole |   jump |   quadpendulum | quadpendulum_theta   | trot   |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|--------|
| trajax   |      6.78 |       3200 | -             |       5.57 |   3200 |           2.52 | -                    | -      |

