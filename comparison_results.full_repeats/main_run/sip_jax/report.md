# LIPA solver comparison

## cartpole

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes         |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-jax  |     231 |      2378.6 |         66.8 |   1.38e-07 |                   0 |   3.16e-08 |   7.4e-08 |  1.38e-07 |           0 |          0 |   1.01e-07 |   9.34e-07 | 9.34e-07 | ok   | Status.SOLVED |


## acrobot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   |   kkt:stat |      KKT | ok   | notes         |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-jax  |      68 |       396.4 |        44.51 |   3.29e-13 |                   0 |   5.93e-15 |  3.29e-13 |         0 |           0 | -          | -          |   9.43e-07 | 9.43e-07 | ok   | Status.SOLVED |


## quadpendulum

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes         |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-jax  |     123 |      4164.3 |        8.279 |   9.63e-11 |                   0 |   1.42e-11 |  9.63e-11 |  1.11e-11 |           0 |          0 |   1.69e-08 |   7.86e-07 | 7.86e-07 | ok   | Status.SOLVED |


## quadpendulum_theta

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes         |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-jax  |     198 |      7456.7 |        6.907 |    8.2e-12 |                   0 |   3.11e-15 |   8.2e-12 |  2.22e-15 |           0 |          0 |   1.92e-08 |   9.31e-07 | 9.31e-07 | ok   | Status.SOLVED |


## barrel_roll

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes         |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------------|
| sip-jax  |     186 |      102198 |        22310 |      4e-07 |               8e-07 |      4e-07 |     4e-07 |         0 |       8e-07 |          0 |        0.1 |   0.000993 |   0.1 | ok   | Status.SOLVED |


## backflip

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                        |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------------------|
| sip-jax  |       0 | 2.40073e+06 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 2400s (subprocess timeout) |


## jump

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT   | ok   | notes                                        |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------------------------------|
| sip-jax  |       0 |  2.4006e+06 |          nan |        nan |                 nan | -          | -         | -         | -           | -          | -          | -          | -     | x    | hard-killed after 2400s (subprocess timeout) |


## trot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |    KKT | ok   | notes         |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|--------|------|---------------|
| sip-jax  |      55 |     25622.7 |        30910 |   4.01e-07 |            7.99e-07 |   4.01e-07 |  4.01e-07 |         0 |    7.99e-07 |          0 |     0.0381 |   0.000971 | 0.0381 | ok   | Status.SOLVED |


## Summary: iterations + status

| solver   | acrobot   | backflip   | barrel_roll   | cartpole   | jump   | quadpendulum   | quadpendulum_theta   | trot   |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|--------|
| sip-jax  | 68 ok     | 0 x        | 186 ok        | 231 ok     | 0 x    | 123 ok         | 198 ok               | 55 ok  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver   | acrobot   | backflip   | barrel_roll   | cartpole   | jump   | quadpendulum   | quadpendulum_theta   | trot        |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|-------------|
| sip-jax  | 396 ms ok | -          | 102199 ms ok  | 2379 ms ok | -      | 4164 ms ok     | 7457 ms ok           | 25623 ms ok |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver   |   acrobot | backflip   |   barrel_roll |   cartpole | jump   |   quadpendulum |   quadpendulum_theta |   trot |
|----------|-----------|------------|---------------|------------|--------|----------------|----------------------|--------|
| sip-jax  |  9.43e-07 | -          |           0.1 |   9.34e-07 | -      |       7.86e-07 |             9.31e-07 | 0.0381 |

