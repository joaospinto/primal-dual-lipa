# LIPA solver comparison

## barrel_roll

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------|
| lipa-gpu |      70 |      3511.6 |        22310 |   0.000152 |            9.12e-07 |   1.76e-07 |  0.000152 |         0 |    9.12e-07 |          0 |     0.0894 |       4.42 |  4.42 | ok   | two-phase warm start |


## backflip

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|---------|
| lipa-gpu |     476 |     29093.6 |        61950 |   8.06e-05 |            2.02e-06 |   8.93e-10 |  8.06e-05 |         0 |    2.02e-06 |          0 |      0.082 |      12700 | 12700 | ok   |         |


## jump

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------|
| lipa-gpu |     223 |     13061.6 |        20920 |    2.2e-06 |             0.00073 |   1.09e-10 |   2.2e-06 |         0 |     0.00073 |          0 |    0.00495 |       1210 |  1210 | ok   | two-phase warm start |


## trot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |   KKT | ok   | notes                |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|-------|------|----------------------|
| lipa-gpu |      29 |        1347 |        30910 |   4.29e-05 |               8e-07 |      4e-07 |  4.29e-05 |         0 |       8e-07 |          0 |      0.753 |       1.94 |  1.94 | ok   | two-phase warm start |


## Summary: iterations + status

| solver   | backflip   | barrel_roll   | jump   | trot   |
|----------|------------|---------------|--------|--------|
| lipa-gpu | 476 ok     | 70 ok         | 223 ok | 29 ok  |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver   | backflip    | barrel_roll   | jump        | trot       |
|----------|-------------|---------------|-------------|------------|
| lipa-gpu | 29094 ms ok | 3512 ms ok    | 13062 ms ok | 1347 ms ok |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver   |   backflip |   barrel_roll |   jump |   trot |
|----------|------------|---------------|--------|--------|
| lipa-gpu |      12700 |          4.42 |   1210 |   1.94 |

