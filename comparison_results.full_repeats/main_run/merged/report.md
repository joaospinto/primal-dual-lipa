# LIPA solver comparison

## cartpole

| solver          |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                                             |
|-----------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------------------------------------------|
| lipa-gpu        |      84 |       133.6 |         66.8 |   8.88e-16 |            0        | 3.64e-22   | 8.88e-16  | 6.90e-21  | 0.00e+00    | 0.00e+00   | 1.01e-13   | 5.81e-09   | 5.81e-09 | ok   |                                                   |
| lipa-cpu        |      82 |        91.5 |         66.8 |   1.33e-15 |            0        | 5.65e-22   | 1.33e-15  | 1.42e-22  | 0.00e+00    | 0.00e+00   | 4.31e-15   | 9.01e-09   | 9.01e-09 | ok   |                                                   |
| ipopt-casadi    |      33 |        58.2 |         66.8 |   2.51e-10 |            0        | 3.41e-34   | 2.51e-10  | 3.74e-37  | 0.00e+00    | 0.00e+00   | 9.82e-08   | 9.15e-09   | 9.82e-08 | ok   | Solve_Succeeded                                   |
| ipopt-jax       |      46 |      5638.5 |         66.8 |   3.4e-11  |            0        | 1.88e-30   | 3.40e-11  | 2.85e-35  | 0.00e+00    | 1.27e+00   | 3.59e+01   | 6.68e+00   | 3.59e+01 | ok   | Solve_Succeeded                                   |
| acados          |      68 |        57.3 |         66.8 |   5.33e-15 |            0        | 0.00e+00   | 5.33e-15  | 4.44e-16  | 0.00e+00    | -          | -          | -          | 5.33e-15 | ok   | status=0                                          |
| fatrop-casadi   |      80 |        41.4 |         66.8 |   1.94e-12 |            0        | 0.00e+00   | 1.94e-12  | 9.83e-24  | 0.00e+00    | 0.00e+00   | 1.78e-07   | 2.83e-11   | 1.78e-07 | ok   | 0                                                 |
| fatrop-jax      |      99 |     37155   |         66.8 |   4.74e-11 |            0        | 0.00e+00   | 4.74e-11  | 2.65e-23  | 0.00e+00    | 0.00e+00   | 1.06e-07   | 4.31e-10   | 1.06e-07 | ok   | 0                                                 |
| sip-casadi      |     231 |      1050.9 |         66.8 |   1.38e-07 |            0        | 3.16e-08   | 7.41e-08  | 1.38e-07  | 0.00e+00    | -          | -          | -          | 1.38e-07 | ok   | Status.SOLVED                                     |
| sip-jax         |     231 |      2378.6 |         66.8 |   1.38e-07 |            0        | 3.16e-08   | 7.40e-08  | 1.38e-07  | 0.00e+00    | 0.00e+00   | 1.01e-07   | 9.34e-07   | 9.34e-07 | ok   | Status.SOLVED                                     |
| csqp-casadi     |     115 |      3929.1 |         66.8 |   4.17e-07 |            0        | 0.00e+00   | 8.88e-16  | 4.17e-07  | 0.00e+00    | 0.00e+00   | 6.72e-10   | 3.23e+01   | 3.23e+01 | ok   |                                                   |
| csqp-jax        |     113 |     22325   |         66.8 |   4.17e-07 |            0        | 0.00e+00   | 1.78e-15  | 4.17e-07  | 0.00e+00    | 0.00e+00   | 6.72e-10   | 3.23e+01   | 3.23e+01 | ok   |                                                   |
| aligator-casadi |     101 |      1299.5 |         66.8 |   1.16e-09 |            4.26e-14 | -          | -         | -         | -           | -          | -          | -          | -        | ok   |                                                   |
| aligator-jax    |     114 |     16403.8 |         66.8 |   1.16e-09 |            4.26e-14 | -          | -         | -         | -           | -          | -          | -          | -        | ok   |                                                   |
| trajax          |     310 |      5396.1 |         66.8 |   1.33e-07 |            0        | 0.00e+00   | 8.88e-16  | 1.33e-07  | 0.00e+00    | 0.00e+00   | 3.67e-14   | 5.57e+00   | 5.57e+00 | ok   | al_iters=8 ilqr_iters_total=310 max_viol=1.33e-07 |


## acrobot

| solver          |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                            |
|-----------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|----------------------------------|
| lipa-gpu        |     108 |       135.9 |        44.51 |   9.23e-11 |                   0 | 2.28e-11   | 9.23e-11  | 0.00e+00  | 0.00e+00    | 0.00e+00   | 0.00e+00   | 3.87e-09   | 3.87e-09 | ok   |                                  |
| lipa-cpu        |     108 |        50.7 |        44.51 |   9.23e-11 |                   0 | 2.28e-11   | 9.23e-11  | 0.00e+00  | 0.00e+00    | 0.00e+00   | 0.00e+00   | 3.87e-09   | 3.87e-09 | ok   |                                  |
| ipopt-casadi    |      21 |        33.5 |        44.51 |   5.48e-07 |                   0 | 6.31e-30   | 5.48e-07  | 0.00e+00  | 0.00e+00    | 0.00e+00   | 0.00e+00   | 5.34e-07   | 5.48e-07 | ok   | Solve_Succeeded                  |
| ipopt-jax       |      21 |      1924.5 |        44.51 |   5.48e-07 |                   0 | 2.84e-29   | 5.48e-07  | 0.00e+00  | 0.00e+00    | -          | -          | 5.35e-07   | 5.48e-07 | ok   | Solve_Succeeded                  |
| acados          |      99 |        11.8 |        44.51 |   3.68e-13 |                   0 | 0.00e+00   | 3.68e-13  | 0.00e+00  | 0.00e+00    | -          | -          | 1.36e+01   | 1.36e+01 | ok   | status=0                         |
| fatrop-casadi   |      17 |        30.3 |        44.51 |   3.16e-11 |                   0 | 0.00e+00   | 3.16e-11  | 0.00e+00  | 0.00e+00    | -          | -          | 3.19e-11   | 3.19e-11 | ok   | 0                                |
| fatrop-jax      |      17 |      4265.3 |        44.51 |   3.16e-11 |                   0 | 0.00e+00   | 3.16e-11  | 0.00e+00  | 0.00e+00    | -          | -          | 3.19e-11   | 3.19e-11 | ok   | 0                                |
| sip-casadi      |      68 |       298.2 |        44.51 |   3.3e-13  |                   0 | 5.93e-15   | 3.30e-13  | 0.00e+00  | 0.00e+00    | 0.00e+00   | 0.00e+00   | 9.44e-07   | 9.44e-07 | ok   | Status.SOLVED                    |
| sip-jax         |      68 |       396.4 |        44.51 |   3.29e-13 |                   0 | 5.93e-15   | 3.29e-13  | 0.00e+00  | 0.00e+00    | -          | -          | 9.43e-07   | 9.43e-07 | ok   | Status.SOLVED                    |
| csqp-casadi     |      74 |       569.5 |        44.51 |   5.33e-15 |                   0 | 0.00e+00   | 5.33e-15  | 0.00e+00  | 0.00e+00    | 0.00e+00   | 0.00e+00   | 1.36e+01   | 1.36e+01 | ok   |                                  |
| csqp-jax        |      74 |      7516.5 |        44.51 |   4.88e-15 |                   0 | 0.00e+00   | 4.88e-15  | 0.00e+00  | 0.00e+00    | 0.00e+00   | 0.00e+00   | 1.36e+01   | 1.36e+01 | ok   |                                  |
| aligator-casadi |      55 |       514.8 |        44.51 |   4.24e-08 |                   0 | -          | -         | -         | -           | -          | -          | -          | -        | ok   |                                  |
| aligator-jax    |      55 |      4988.9 |        44.51 |   4.24e-08 |                   0 | -          | -         | -         | -           | -          | -          | -          | -        | ok   |                                  |
| trajax          |      86 |       876.4 |        44.51 |   4.44e-16 |                   0 | 0.00e+00   | 4.44e-16  | 0.00e+00  | 0.00e+00    | -          | -          | 6.78e+00   | 6.78e+00 | ok   | ilqr_iters=86 grad_norm=8.61e-07 |


## quadpendulum

| solver          |   iters |        time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                                             |
|-----------------|---------|------------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------------------------------------------|
| lipa-gpu        |      81 |    170.5         |        8.279 |   1.33e-07 |            0        | 9.42e-09   | 1.33e-07  | 1.36e-09  | 0.00e+00    | 0.00e+00   | 1.24e-07   | 4.06e-05   | 4.06e-05 | ok   |                                                   |
| lipa-cpu        |      81 |    668.2         |        8.279 |   1.33e-07 |            0        | 9.42e-09   | 1.33e-07  | 1.36e-09  | 0.00e+00    | 0.00e+00   | 1.24e-07   | 4.06e-05   | 4.06e-05 | ok   |                                                   |
| ipopt-casadi    |      65 |   4000.4         |        8.279 |   3.48e-12 |            1e-08    | 4.95e-31   | 3.48e-12  | 1.37e-32  | 1.00e-08    | 0.00e+00   | 1.97e-08   | 2.21e-11   | 1.97e-08 | ok   | Solve_Succeeded                                   |
| ipopt-jax       |     163 |  95391.4         |       11.75  |   2.07e-08 |            0        | 5.86e-24   | 2.07e-08  | 1.65e-27  | 0.00e+00    | 2.43e-01   | 1.49e+01   | 9.37e+00   | 1.49e+01 | ok   | Solve_Succeeded                                   |
| acados          |      31 |    527           |       11.75  |   1.2e-11  |            0        | 0.00e+00   | 1.20e-11  | 1.39e-12  | 0.00e+00    | -          | -          | -          | 1.20e-11 | ok   | status=0                                          |
| fatrop-casadi   |     112 |   1595.5         |        8.279 |   8.38e-12 |            0        | 0.00e+00   | 8.38e-12  | 6.21e-21  | 0.00e+00    | 0.00e+00   | 1.00e-07   | 3.45e-11   | 1.00e-07 | ok   | 0                                                 |
| fatrop-jax      |     112 | 189991           |        8.279 |   8.38e-12 |            0        | 0.00e+00   | 8.38e-12  | 3.43e-21  | 0.00e+00    | 0.00e+00   | 1.00e-07   | 3.45e-11   | 1.00e-07 | ok   | 0                                                 |
| sip-casadi      |     231 |   6328.4         |        8.279 |   6.19e-12 |            0        | 2.44e-15   | 6.19e-12  | 3.11e-15  | 0.00e+00    | 0.00e+00   | 4.85e-08   | -          | 4.85e-08 | ok   | Status.SOLVED                                     |
| sip-jax         |     123 |   4164.3         |        8.279 |   9.63e-11 |            0        | 1.42e-11   | 9.63e-11  | 1.11e-11  | 0.00e+00    | 0.00e+00   | 1.69e-08   | 7.86e-07   | 7.86e-07 | ok   | Status.SOLVED                                     |
| csqp-casadi     |    1000 | 315738           |        8.694 |   1.82e-06 |            8.03e-05 | 0.00e+00   | 1.82e-06  | 7.08e-09  | 8.03e-05    | 0.00e+00   | 1.88e-04   | 6.86e+00   | 6.86e+00 | x    |                                                   |
| csqp-jax        |    1000 |      1.07741e+06 |        8.694 |   1.87e-06 |            7.84e-05 | 0.00e+00   | 1.87e-06  | 6.99e-09  | 7.84e-05    | 2.67e-23   | 1.83e-04   | 6.85e+00   | 6.85e+00 | x    |                                                   |
| aligator-casadi |    1000 | 154943           |       22.45  |   0.0199   |            0.0398   | -          | -         | -         | -           | -          | -          | -          | -        | x    |                                                   |
| aligator-jax    |       0 | 360529           |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 360s (subprocess timeout)       |
| trajax          |     237 |  12248.3         |        9.354 |   1.78e-08 |            6.15e-07 | 0.00e+00   | 8.88e-16  | 1.78e-08  | 6.15e-07    | 0.00e+00   | 7.26e-07   | 2.52e+00   | 2.52e+00 | ok   | al_iters=5 ilqr_iters_total=237 max_viol=6.15e-07 |


## quadpendulum_theta

| solver          |   iters |   time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                                                            |
|-----------------|---------|-------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|------------------------------------------------------------------|
| lipa-gpu        |     143 |       351.3 |        7.269 |   6.89e-09 |            0        | 1.76e-10   | 6.89e-09  | 1.70e-10  | 0.00e+00    | 0.00e+00   | 1.81e-07   | 4.88e-05   | 4.88e-05 | ok   |                                                                  |
| lipa-cpu        |     143 |      1357.6 |        7.269 |   6.89e-09 |            0        | 1.76e-10   | 6.89e-09  | 1.70e-10  | 0.00e+00    | 0.00e+00   | 1.81e-07   | 4.88e-05   | 4.88e-05 | ok   |                                                                  |
| ipopt-casadi    |     103 |      6768.9 |        6.907 |   1.68e-09 |            0        | 4.58e-26   | 1.68e-09  | 2.31e-31  | 0.00e+00    | 0.00e+00   | 1.04e-07   | 1.06e-07   | 1.06e-07 | ok   | Solve_Succeeded                                                  |
| ipopt-jax       |      82 |     51383.3 |       14.81  |   2.62e-07 |            8.24e-09 | 2.77e-28   | 2.62e-07  | 4.45e-31  | 8.24e-09    | 1.69e-01   | 2.40e+02   | 1.74e+02   | 2.40e+02 | ok   | Solve_Succeeded                                                  |
| acados          |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | acados does not natively support cross-stage Theta (theta_dim=1) |
| fatrop-casadi   |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | fatrop does not natively support cross-stage Theta (theta_dim=1) |
| fatrop-jax      |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | fatrop does not natively support cross-stage Theta (theta_dim=1) |
| sip-casadi      |     198 |      5725.5 |        6.907 |   8.2e-12  |            0        | 3.11e-15   | 8.20e-12  | 2.22e-15  | 0.00e+00    | 0.00e+00   | 1.92e-08   | -          | 1.92e-08 | ok   | Status.SOLVED                                                    |
| sip-jax         |     198 |      7456.7 |        6.907 |   8.2e-12  |            0        | 3.11e-15   | 8.20e-12  | 2.22e-15  | 0.00e+00    | 0.00e+00   | 1.92e-08   | 9.31e-07   | 9.31e-07 | ok   | Status.SOLVED                                                    |
| csqp-casadi     |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | CSQP/Crocoddyl does not support cross-stage Theta (theta_dim=1)  |
| csqp-jax        |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | CSQP/Crocoddyl does not support cross-stage Theta (theta_dim=1)  |
| aligator-casadi |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | aligator does not support cross-stage Theta (theta_dim=1)        |
| aligator-jax    |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | aligator does not support cross-stage Theta (theta_dim=1)        |
| trajax          |       0 |         0   |      nan     | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | trajax does not natively support cross-stage Theta (theta_dim=1) |


## barrel_roll

| solver     |   iters |        time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                                        |
|------------|---------|------------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|----------------------------------------------|
| lipa-gpu   |      70 |   3253.9         |        22310 |   0.000152 |            9.12e-07 | 1.76e-07   | 1.52e-04  | 0.00e+00  | 9.12e-07    | 0.00e+00   | 8.94e-02   | 4.42e+00   | 4.42e+00 | ok   | two-phase warm start                         |
| ipopt-jax  |     388 |      1.20163e+06 |        26360 |   3.16     |            8.02e-07 | 3.96e-07   | 3.16e+00  | 0.00e+00  | 8.02e-07    | 1.38e+00   | 9.92e+01   | 1.11e+02   | 1.11e+02 | x    | Maximum_WallTime_Exceeded                    |
| fatrop-jax |       0 |      2.4004e+06  |          nan | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 2400s (subprocess timeout) |
| sip-jax    |     186 | 102198           |        22310 |   4e-07    |            8e-07    | 4.00e-07   | 4.00e-07  | 0.00e+00  | 8.00e-07    | 0.00e+00   | 1.00e-01   | 9.93e-04   | 1.00e-01 | ok   | Status.SOLVED                                |
| csqp-jax   |     200 | 352011           |        22230 |   0.198    |           26.9      | 0.00e+00   | 1.98e-01  | 0.00e+00  | 2.69e+01    | 5.55e-17   | 2.31e+01   | 2.77e+05   | 2.77e+05 | x    |                                              |
| trajax     |       0 |      2.40051e+06 |          nan | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 2400s (subprocess timeout) |


## backflip

| solver     |   iters |        time [ms] |    final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                                             |
|------------|---------|------------------|---------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|---------------------------------------------------|
| lipa-gpu   |     476 |  31689           | 61950         |   8.06e-05 |            2.02e-06 | 8.93e-10   | 8.06e-05  | 0.00e+00  | 2.02e-06    | 0.00e+00   | 8.20e-02   | 1.27e+04   | 1.27e+04 | ok   |                                                   |
| ipopt-jax  |     211 |      1.20247e+06 |     1.626e+06 |   1.7      |            0.959    | 4.76e-01   | 1.70e+00  | 0.00e+00  | 9.59e-01    | 2.56e+03   | 6.53e+03   | 1.53e+06   | 1.53e+06 | x    | Maximum_WallTime_Exceeded                         |
| fatrop-jax |       0 |      2.40053e+06 |   nan         | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 2400s (subprocess timeout)      |
| sip-jax    |       0 |      2.40073e+06 |   nan         | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 2400s (subprocess timeout)      |
| csqp-jax   |     200 | 656878           | 31460         |   0.0126   |           80.6      | 0.00e+00   | 1.26e-02  | 0.00e+00  | 8.06e+01    | 8.88e-17   | 2.15e+00   | 2.43e+04   | 2.43e+04 | x    |                                                   |
| trajax     |     196 | 240760           | 41740         |   8.88e-16 |            2.08e-05 | 0.00e+00   | 8.88e-16  | 0.00e+00  | 2.08e-05    | 0.00e+00   | 5.66e-04   | 3.20e+03   | 3.20e+03 | ok   | al_iters=2 ilqr_iters_total=196 max_viol=2.08e-05 |


## jump

| solver     |   iters |        time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                                                                   |
|------------|---------|------------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|-------------------------------------------------------------------------|
| lipa-gpu   |     223 |  12963.6         |        20920 |   2.2e-06  |            0.00073  | 1.09e-10   | 2.20e-06  | 0.00e+00  | 7.30e-04    | 0.00e+00   | 4.95e-03   | 1.21e+03   | 1.21e+03 | ok   | two-phase warm start                                                    |
| ipopt-jax  |     285 |      1.20348e+06 |       709500 |   0.0101   |            0.0366   | 1.75e-04   | 1.01e-02  | 0.00e+00  | 3.66e-02    | 9.90e+02   | 6.81e+04   | 5.53e+04   | 6.81e+04 | x    | Maximum_WallTime_Exceeded                                               |
| fatrop-jax |       0 |      2.4004e+06  |          nan | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 2400s (subprocess timeout)                            |
| sip-jax    |       0 |      2.4006e+06  |          nan | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 2400s (subprocess timeout)                            |
| csqp-jax   |     400 | 948528           |        19500 |   0.653    |          103        | 0.00e+00   | 6.53e-01  | 0.00e+00  | 1.03e+02    | 0.00e+00   | 1.60e+00   | 2.92e+04   | 2.92e+04 | x    | two-phase warm start                                                    |
| trajax     |    1027 |      1.04363e+06 |        19640 |   1.68e-15 |            0.000374 | 0.00e+00   | 1.68e-15  | 0.00e+00  | 3.74e-04    | 0.00e+00   | 1.40e-04   | 3.20e+03   | 3.20e+03 | ok   | two-phase warm start; al_iters=1 ilqr_iters_total=977 max_viol=3.74e-04 |


## trot

| solver     |   iters |        time [ms] |   final cost |   |eq|_inf |   |max(0,ineq)|_inf | kkt:init   | kkt:dyn   | kkt:eq*   | kkt:ineq*   | kkt:dual   | kkt:comp   | kkt:stat   | KKT      | ok   | notes                                        |
|------------|---------|------------------|--------------|------------|---------------------|------------|-----------|-----------|-------------|------------|------------|------------|----------|------|----------------------------------------------|
| lipa-gpu   |      29 |   1469.8         |        30910 |   4.29e-05 |            8e-07    | 4.00e-07   | 4.29e-05  | 0.00e+00  | 8.00e-07    | 0.00e+00   | 7.53e-01   | 1.94e+00   | 1.94e+00 | ok   | two-phase warm start                         |
| ipopt-jax  |     117 | 324954           |        30910 |   3.96e-07 |            8.02e-07 | 3.96e-07   | 3.96e-07  | 0.00e+00  | 8.02e-07    | 0.00e+00   | 4.38e+00   | 4.00e-05   | 4.38e+00 | ok   | Solve_Succeeded                              |
| fatrop-jax |     124 |      1.49462e+06 |        31030 |   3.02e-06 |            1e-06    | 0.00e+00   | 3.02e-06  | 0.00e+00  | 1.00e-06    | 0.00e+00   | 6.64e+13   | 7.44e+03   | 6.64e+13 | ok   | RuntimeError: return_status='1'; 1           |
| sip-jax    |      55 |  25622.7         |        30910 |   4.01e-07 |            7.99e-07 | 4.01e-07   | 4.01e-07  | 0.00e+00  | 7.99e-07    | 0.00e+00   | 3.81e-02   | 9.71e-04   | 3.81e-02 | ok   | Status.SOLVED                                |
| csqp-jax   |     212 | 312338           |        30900 |   0.131    |            2.96     | 0.00e+00   | 1.31e-01  | 0.00e+00  | 2.96e+00    | 0.00e+00   | 5.45e+00   | 2.77e+05   | 2.77e+05 | x    | two-phase warm start                         |
| trajax     |       0 |      2.4006e+06  |          nan | nan        |          nan        | -          | -         | -         | -           | -          | -          | -          | -        | x    | hard-killed after 2400s (subprocess timeout) |


## Summary: iterations + status

| solver          | acrobot   | backflip   | barrel_roll   | cartpole   | jump    | quadpendulum   | quadpendulum_theta   | trot   |
|-----------------|-----------|------------|---------------|------------|---------|----------------|----------------------|--------|
| acados          | 99 ok     | gated      | gated         | 68 ok      | gated   | 31 ok          | N/A                  | gated  |
| aligator-casadi | 55 ok     | gated      | gated         | 101 ok     | gated   | 1000 x         | N/A                  | gated  |
| aligator-jax    | 55 ok     | gated      | gated         | 114 ok     | gated   | 0 x            | N/A                  | gated  |
| csqp-casadi     | 74 ok     | gated      | gated         | 115 ok     | gated   | 1000 x         | N/A                  | gated  |
| csqp-jax        | 74 ok     | 200 x      | 200 x         | 113 ok     | 400 x   | 1000 x         | N/A                  | 212 x  |
| fatrop-casadi   | 17 ok     | gated      | gated         | 80 ok      | gated   | 112 ok         | N/A                  | gated  |
| fatrop-jax      | 17 ok     | 0 x        | 0 x           | 99 ok      | 0 x     | 112 ok         | N/A                  | 124 ok |
| ipopt-casadi    | 21 ok     | gated      | gated         | 33 ok      | gated   | 65 ok          | 103 ok               | gated  |
| ipopt-jax       | 21 ok     | 211 x      | 388 x         | 46 ok      | 285 x   | 163 ok         | 82 ok                | 117 ok |
| lipa-cpu        | 108 ok    | gated      | gated         | 82 ok      | gated   | 81 ok          | 143 ok               | gated  |
| lipa-gpu        | 108 ok    | 476 ok     | 70 ok         | 84 ok      | 223 ok  | 81 ok          | 143 ok               | 29 ok  |
| sip-casadi      | 68 ok     | gated      | gated         | 231 ok     | gated   | 231 ok         | 198 ok               | gated  |
| sip-jax         | 68 ok     | 0 x        | 186 ok        | 231 ok     | 0 x     | 123 ok         | 198 ok               | 55 ok  |
| trajax          | 86 ok     | 196 ok     | 0 x           | 310 ok     | 1027 ok | 237 ok         | N/A                  | 0 x    |


## Summary: wall-clock time (excludes JIT / codegen / one-time setup)

| solver          | acrobot    | backflip     | barrel_roll   | cartpole    | jump          | quadpendulum   | quadpendulum_theta   | trot          |
|-----------------|------------|--------------|---------------|-------------|---------------|----------------|----------------------|---------------|
| acados          | 12 ms ok   | -            | -             | 57 ms ok    | -             | 527 ms ok      | -                    | -             |
| aligator-casadi | 515 ms ok  | -            | -             | 1299 ms ok  | -             | 154943 ms x    | -                    | -             |
| aligator-jax    | 4989 ms ok | -            | -             | 16404 ms ok | -             | -              | -                    | -             |
| csqp-casadi     | 570 ms ok  | -            | -             | 3929 ms ok  | -             | 315738 ms x    | -                    | -             |
| csqp-jax        | 7517 ms ok | 656878 ms x  | 352011 ms x   | 22325 ms ok | 948528 ms x   | 1077415 ms x   | -                    | 312337 ms x   |
| fatrop-casadi   | 30 ms ok   | -            | -             | 41 ms ok    | -             | 1596 ms ok     | -                    | -             |
| fatrop-jax      | 4265 ms ok | -            | -             | 37155 ms ok | -             | 189991 ms ok   | -                    | 1494616 ms ok |
| ipopt-casadi    | 34 ms ok   | -            | -             | 58 ms ok    | -             | 4000 ms ok     | 6769 ms ok           | -             |
| ipopt-jax       | 1924 ms ok | 1202465 ms x | 1201628 ms x  | 5638 ms ok  | 1203482 ms x  | 95391 ms ok    | 51383 ms ok          | 324954 ms ok  |
| lipa-cpu        | 51 ms ok   | -            | -             | 92 ms ok    | -             | 668 ms ok      | 1358 ms ok           | -             |
| lipa-gpu        | 136 ms ok  | 31689 ms ok  | 3254 ms ok    | 134 ms ok   | 12964 ms ok   | 171 ms ok      | 351 ms ok            | 1470 ms ok    |
| sip-casadi      | 298 ms ok  | -            | -             | 1051 ms ok  | -             | 6328 ms ok     | 5725 ms ok           | -             |
| sip-jax         | 396 ms ok  | -            | 102199 ms ok  | 2379 ms ok  | -             | 4164 ms ok     | 7457 ms ok           | 25623 ms ok   |
| trajax          | 876 ms ok  | 240760 ms ok | -             | 5396 ms ok  | 1043632 ms ok | 12248 ms ok    | -                    | -             |


## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)

| solver          | acrobot   | backflip   | barrel_roll   | cartpole   | jump     | quadpendulum   | quadpendulum_theta   | trot     |
|-----------------|-----------|------------|---------------|------------|----------|----------------|----------------------|----------|
| acados          | 1.36e+01  | -          | -             | 5.33e-15   | -        | 1.20e-11       | -                    | -        |
| aligator-casadi | -         | -          | -             | -          | -        | -              | -                    | -        |
| aligator-jax    | -         | -          | -             | -          | -        | -              | -                    | -        |
| csqp-casadi     | 1.36e+01  | -          | -             | 3.23e+01   | -        | 6.86e+00       | -                    | -        |
| csqp-jax        | 1.36e+01  | 2.43e+04   | 2.77e+05      | 3.23e+01   | 2.92e+04 | 6.85e+00       | -                    | 2.77e+05 |
| fatrop-casadi   | 3.19e-11  | -          | -             | 1.78e-07   | -        | 1.00e-07       | -                    | -        |
| fatrop-jax      | 3.19e-11  | -          | -             | 1.06e-07   | -        | 1.00e-07       | -                    | 6.64e+13 |
| ipopt-casadi    | 5.48e-07  | -          | -             | 9.82e-08   | -        | 1.97e-08       | 1.06e-07             | -        |
| ipopt-jax       | 5.48e-07  | 1.53e+06   | 1.11e+02      | 3.59e+01   | 6.81e+04 | 1.49e+01       | 2.40e+02             | 4.38e+00 |
| lipa-cpu        | 3.87e-09  | -          | -             | 9.01e-09   | -        | 4.06e-05       | 4.88e-05             | -        |
| lipa-gpu        | 3.87e-09  | 1.27e+04   | 4.42e+00      | 5.81e-09   | 1.21e+03 | 4.06e-05       | 4.88e-05             | 1.94e+00 |
| sip-casadi      | 9.44e-07  | -          | -             | 1.38e-07   | -        | 4.85e-08       | 1.92e-08             | -        |
| sip-jax         | 9.43e-07  | -          | 1.00e-01      | 9.34e-07   | -        | 7.86e-07       | 9.31e-07             | 3.81e-02 |
| trajax          | 6.78e+00  | 3.20e+03   | -             | 5.57e+00   | 3.20e+03 | 2.52e+00       | -                    | -        |

