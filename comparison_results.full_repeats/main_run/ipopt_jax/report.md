# LIPA solver comparison

## barrel_roll

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                     |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------------------------|
| ipopt-jax |     388 | 1.20163e+06 |        26360 |       3.16 |            8.02e-07 |   3.96e-07 |      3.16 |         0 |    8.02e-07 |       1.38 |       99.2 |        111 |   111 | x    | Maximum_WallTime_Exceeded |


## backflip

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes                     |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------------------|
| ipopt-jax |     211 | 1.20247e+06 |    1.626e+06 |        1.7 |               0.959 |      0.476 |       1.7 |         0 |       0.959 |       2560 |       6530 |   1.53e+06 | 1.53e+06 | x    | Maximum_WallTime_Exceeded |


## jump

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                     |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------------------------|
| ipopt-jax |     285 | 1.20348e+06 |       709500 |     0.0101 |              0.0366 |   0.000175 |    0.0101 |         0 |      0.0366 |        990 |      68100 |      55300 | 68100 | x    | Maximum_WallTime_Exceeded |


## trot

| solver    |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes           |
|-----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|-----------------|
| ipopt-jax |     117 |      324954 |        30910 |   3.96e-07 |            8.02e-07 |   3.96e-07 |  3.96e-07 |         0 |    8.02e-07 |          0 |       4.38 |      4e-05 |  4.38 | ok   | Solve_Succeeded |


## Summary: iterations + status

| solver    | backflip   | barrel_roll   | jump   | trot   |
|-----------|------------|---------------|--------|--------|
| ipopt-jax | 211 x      | 388 x         | 285 x  | 117 ok |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver    | backflip     | barrel_roll   | jump         | trot         |
|-----------|--------------|---------------|--------------|--------------|
| ipopt-jax | 1202465 ms x | 1201628 ms x  | 1203482 ms x | 324954 ms ok |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver    |   backflip |   barrel_roll |   jump |   trot |
|-----------|------------|---------------|--------|--------|
| ipopt-jax |   1.53e+06 |           111 |  68100 |   4.38 |

