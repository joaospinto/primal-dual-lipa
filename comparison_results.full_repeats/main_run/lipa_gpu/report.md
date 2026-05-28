# LIPA solver comparison

## cartpole

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-gpu |      84 |       133.6 |         66.8 |   8.88e-16 |                   0 |   3.64e-22 |  8.88e-16 |   6.9e-21 |           0 |          0 |   1.01e-13 |   5.81e-09 | 5.81e-09 | ok   |         |


## acrobot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-gpu |     108 |       135.9 |        44.51 |   9.23e-11 |                   0 |   2.28e-11 |  9.23e-11 |         0 |           0 |          0 |          0 |   3.87e-09 | 3.87e-09 | ok   |         |


## quadpendulum

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-gpu |      81 |       170.5 |        8.279 |   1.33e-07 |                   0 |   9.42e-09 |  1.33e-07 |  1.36e-09 |           0 |          0 |   1.24e-07 |   4.06e-05 | 4.06e-05 | ok   |         |


## quadpendulum_theta

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-gpu |     143 |       351.3 |        7.269 |   6.89e-09 |                   0 |   1.76e-10 |  6.89e-09 |   1.7e-10 |           0 |          0 |   1.81e-07 |   4.88e-05 | 4.88e-05 | ok   |         |


## barrel_roll

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------|
| lipa-gpu |      70 |      3253.9 |        22310 |   0.000152 |            9.12e-07 |   1.76e-07 |  0.000152 |         0 |    9.12e-07 |          0 |     0.0894 |       4.42 |  4.42 | ok   | two-phase warm start |


## backflip

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| lipa-gpu |     476 |       31689 |        61950 |   8.06e-05 |            2.02e-06 |   8.93e-10 |  8.06e-05 |         0 |    2.02e-06 |          0 |      0.082 |      12700 | 12700 | ok   |         |


## jump

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------|
| lipa-gpu |     223 |     12963.6 |        20920 |    2.2e-06 |             0.00073 |   1.09e-10 |   2.2e-06 |         0 |     0.00073 |          0 |    0.00495 |       1210 |  1210 | ok   | two-phase warm start |


## trot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------|
| lipa-gpu |      29 |      1469.8 |        30910 |   4.29e-05 |               8e-07 |      4e-07 |  4.29e-05 |         0 |       8e-07 |          0 |      0.753 |       1.94 |  1.94 | ok   | two-phase warm start |


## Summary: iterations + status

| solver   | acrobot   | backflip   | barrel_roll   | cartpole   | jump   | quadpendulum   | quadpendulum_theta   | trot   |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|--------|
| lipa-gpu | 108 ok    | 476 ok     | 70 ok         | 84 ok      | 223 ok | 81 ok          | 143 ok               | 29 ok  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver   | acrobot   | backflip    | barrel_roll   | cartpole   | jump        | quadpendulum   | quadpendulum_theta   | trot       |
|----------|-----------|-------------|---------------|------------|-------------|----------------|----------------------|------------|
| lipa-gpu | 136 ms ok | 31689 ms ok | 3254 ms ok    | 134 ms ok  | 12964 ms ok | 171 ms ok      | 351 ms ok            | 1470 ms ok |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver   |   acrobot |   backflip |   barrel_roll |   cartpole |   jump |   quadpendulum |   quadpendulum_theta |   trot |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|--------|
| lipa-gpu |  3.87e-09 |      12700 |          4.42 |   5.81e-09 |   1210 |       4.06e-05 |             4.88e-05 |   1.94 |

