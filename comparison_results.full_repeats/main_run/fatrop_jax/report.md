# LIPA solver comparison

## barrel_roll

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                        |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------------------|
| fatrop-jax |       0 |  2.4004e+06 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 2400s (subprocess timeout) |


## backflip

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                        |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------------------|
| fatrop-jax |       0 | 2.40053e+06 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 2400s (subprocess timeout) |


## jump

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                        |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------------------|
| fatrop-jax |       0 |  2.4004e+06 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 2400s (subprocess timeout) |


## trot

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes                              |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|------------------------------------|
| fatrop-jax |     124 | 1.49462e+06 |        31030 |   3.02e-06 |               1e-06 |          0 |  3.02e-06 |         0 |       1e-06 |          0 |   6.64e+13 |       7440 | 6.64e+13 | ok   | RuntimeError: return_status='1'; 1 |


## Summary: iterations + status

| solver     | backflip   | barrel_roll   | jump   | trot   |
|------------|------------|---------------|--------|--------|
| fatrop-jax | 0 x        | 0 x           | 0 x    | 124 ok |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver     | backflip   | barrel_roll   | jump   | trot          |
|------------|------------|---------------|--------|---------------|
| fatrop-jax | -          | -             | -      | 1494616 ms ok |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver     | backflip   | barrel_roll   | jump   |     trot |
|------------|------------|---------------|--------|----------|
| fatrop-jax | -          | -             | -      | 6.64e+13 |

