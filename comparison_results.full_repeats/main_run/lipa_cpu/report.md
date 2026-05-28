# LIPA solver comparison

## cartpole

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-cpu |      82 |        91.5 |         66.8 |   1.33e-15 |                   0 |   5.65e-22 |  1.33e-15 |  1.42e-22 |           0 |          0 |   4.31e-15 |   9.01e-09 | 9.01e-09 | ok   |         |


## acrobot

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-cpu |     108 |        50.7 |        44.51 |   9.23e-11 |                   0 |   2.28e-11 |  9.23e-11 |         0 |           0 |          0 |          0 |   3.87e-09 | 3.87e-09 | ok   |         |


## quadpendulum

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-cpu |      81 |       668.2 |        8.279 |   1.33e-07 |                   0 |   9.42e-09 |  1.33e-07 |  1.36e-09 |           0 |          0 |   1.24e-07 |   4.06e-05 | 4.06e-05 | ok   |         |


## quadpendulum_theta

| solver   |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes   |
|----------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------|
| lipa-cpu |     143 |      1357.6 |        7.269 |   6.89e-09 |                   0 |   1.76e-10 |  6.89e-09 |   1.7e-10 |           0 |          0 |   1.81e-07 |   4.88e-05 | 4.88e-05 | ok   |         |


## Summary: iterations + status

| solver   | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|----------|-----------|------------|----------------|----------------------|
| lipa-cpu | 108 ok    | 82 ok      | 81 ok          | 143 ok               |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver   | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|----------|-----------|------------|----------------|----------------------|
| lipa-cpu | 51 ms ok  | 92 ms ok   | 668 ms ok      | 1358 ms ok           |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver   |   acrobot |   cartpole |   quadpendulum |   quadpendulum_theta |
|----------|-----------|------------|----------------|----------------------|
| lipa-cpu |  3.87e-09 |   9.01e-09 |       4.06e-05 |             4.88e-05 |

