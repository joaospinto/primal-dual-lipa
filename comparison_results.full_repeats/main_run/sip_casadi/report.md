# LIPA solver comparison

## cartpole

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* | kkt:dual   | kkt:comp   | kkt:stat   |      KKT | ok   | notes         |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-casadi |     231 |      1050.9 |         66.8 |   1.38e-07 |                   0 |   3.16e-08 |  7.41e-08 |  1.38e-07 |           0 | -          | -          | -          | 1.38e-07 | ok   | Status.SOLVED |


## acrobot

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp |   kkt:stat |      KKT | ok   | notes         |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-casadi |      68 |       298.2 |        44.51 |    3.3e-13 |                   0 |   5.93e-15 |   3.3e-13 |         0 |           0 |          0 |          0 |   9.44e-07 | 9.44e-07 | ok   | Status.SOLVED |


## quadpendulum

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp | kkt:stat   |      KKT | ok   | notes         |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-casadi |     231 |      6328.4 |        8.279 |   6.19e-12 |                   0 |   2.44e-15 |  6.19e-12 |  3.11e-15 |           0 |          0 |   4.85e-08 | -          | 4.85e-08 | ok   | Status.SOLVED |


## quadpendulum_theta

| solver     |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf |   kkt:init |   kkt:dyn |   kkt:eq* |   kkt:ineq* |   kkt:dual |   kkt:comp | kkt:stat   |      KKT | ok   | notes         |
|------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------|
| sip-casadi |     198 |      5725.5 |        6.907 |    8.2e-12 |                   0 |   3.11e-15 |   8.2e-12 |  2.22e-15 |           0 |          0 |   1.92e-08 | -          | 1.92e-08 | ok   | Status.SOLVED |


## Summary: iterations + status

| solver     | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|------------|-----------|------------|----------------|----------------------|
| sip-casadi | 68 ok     | 231 ok     | 231 ok         | 198 ok               |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver     | acrobot   | cartpole   | quadpendulum   | quadpendulum_theta   |
|------------|-----------|------------|----------------|----------------------|
| sip-casadi | 298 ms ok | 1051 ms ok | 6328 ms ok     | 5725 ms ok           |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver     |   acrobot |   cartpole |   quadpendulum |   quadpendulum_theta |
|------------|-----------|------------|----------------|----------------------|
| sip-casadi |  9.44e-07 |   1.38e-07 |       4.85e-08 |             1.92e-08 |

