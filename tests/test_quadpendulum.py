# From https://github.com/google/trajax/blob/main/notebooks/l4dc/QuadPend.ipynb

import unittest
from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from primal_dual_lipa.integrators import euler
from primal_dual_lipa.optimizers import SolverSettings, solve

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def _wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    """Wrap x to lie within [-pi, pi]."""
    # From https://github.com/google/trajax/blob/main/trajax/experimental/sqp/util.py.
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def get_s1_wrapper(s1_ind: tuple[int, ...]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function for wrapping S1 components of state to [-pi, pi]."""
    # From https://github.com/google/trajax/blob/main/trajax/experimental/sqp/util.py.
    idxs = jnp.array(s1_ind)

    def state_wrapper(x: jnp.ndarray) -> jnp.ndarray:
        return x.at[idxs].set(_wrap_to_pi(x[idxs]))

    return jax.jit(state_wrapper)


n = 8
m = 2
T = 160


Mass = 0.486
mass = 0.2 * Mass
grav = 9.81
l = 0.25  # noqa: E741
L = 2.0 * l
J = 0.00383
fric = 0.01

u_hover = 0.5 * (Mass + mass) * grav * jnp.ones((m,))


# State: q = (p_x, p_y, theta, phi, velocities)  # noqa: ERA001
# where theta: rotation angle of quad
# phi: rotation angle of pendulum, w.r.t. vertical (NOTE: not a relative angle)


def get_mass_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Return the mass matrix M_q."""
    phi = q[-1]
    return jnp.array(
        [
            [Mass + mass, 0.0, 0.0, mass * L * jnp.cos(phi)],
            [0.0, Mass + mass, 0.0, mass * L * jnp.sin(phi)],
            [0.0, 0.0, J, 0.0],
            [
                mass * L * jnp.cos(phi),
                mass * L * jnp.sin(phi),
                0.0,
                mass * L * L,
            ],
        ]
    )


def get_mass_inv(q: jnp.ndarray) -> jnp.ndarray:
    """Return the inverse mass matrix M_inv."""
    phi = q[-1]
    a = Mass + mass
    b = mass * L * jnp.cos(phi)
    c = mass * L * jnp.sin(phi)
    d = mass * L * L
    den = (mass * L) ** 2.0 - a * d
    return jnp.array(
        [
            [(c * c - a * d) / (a * den), -(b * c) / (a * den), 0.0, (b / den)],
            [-(b * c) / (a * den), (b * b - a * d) / (a * den), 0.0, (c / den)],
            [0.0, 0.0, (1.0 / J), 0.0],
            [(b / den), (c / den), 0.0, -(a / den)],
        ]
    )


@jax.jit
def ode(x: jnp.ndarray, u: jnp.ndarray, t: jnp.double) -> jnp.ndarray:
    """Provide the dynamics ODE."""

    def kinetic(q: jnp.ndarray, q_dot: jnp.ndarray) -> jnp.ndarray:
        """Define the kinetic energy."""
        return 0.5 * jnp.vdot(q_dot, get_mass_matrix(q) @ q_dot)

    def potential(q: jnp.ndarray) -> jnp.ndarray:
        """Define the potential energy."""
        return Mass * grav * q[1] + mass * grav * (q[1] - L * jnp.cos(q[-1]))

    def lag(q: jnp.ndarray, q_dot: jnp.ndarray) -> jnp.ndarray:
        """Define the physical Lagrangian."""
        return kinetic(q, q_dot) - potential(q)

    del t

    dL_dq = jax.grad(lag, 0)

    q, q_dot = jnp.split(x, [4])
    # (M_q * q_ddot + M_dot * q_dot) - (dL_dq) = F_q
    _unused_M_q, M_dot = jax.jvp(get_mass_matrix, (q,), (q_dot,))
    M_inv = get_mass_inv(q)
    torque_fric_pole = -fric * (q_dot[-1] - q_dot[-2])
    F_q = jnp.array(
        [
            -jnp.sum(u) * jnp.sin(q[2]),
            jnp.sum(u) * jnp.cos(q[2]),
            (u[0] - u[1]) * l - torque_fric_pole,
            torque_fric_pole,
        ]
    )
    q_ddot = M_inv @ (F_q + dL_dq(q, q_dot) - (M_dot @ q_dot))
    return jnp.concatenate((q_dot, q_ddot))


dt = 0.025
dynamics = euler(ode, dt)


def obs_constraint(
    q: jnp.ndarray, obs: list[tuple[jnp.ndarray, jnp.double]]
) -> jnp.ndarray:
    """Define the obstacle constraints."""

    def get_closest_point(
        endp: tuple[jnp.ndarray, jnp.ndarray], p_o: jnp.ndarray
    ) -> jnp.ndarray:
        """Get closest point between point and straight-line between endpoints."""
        x, y = endp
        t_ = jnp.vdot(p_o - x, y - x) / jnp.vdot(y - x, y - x)
        t_min = jnp.minimum(1.0, jnp.maximum(0.0, t_))
        return x + t_min * (y - x)

    pos = q[:2]
    theta = q[2]
    phi = q[-1]

    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    pos_c = pos + R @ jnp.array([0.0, 0.15 * l])
    pole = (pos, pos + jnp.array([L * jnp.sin(phi), -L * jnp.cos(phi)]))

    def avoid_obs(
        pos_c: jnp.ndarray,
        pole: tuple[jnp.ndarray, jnp.ndarray],
        ob: tuple[jnp.ndarray, jnp.double],
    ) -> jnp.ndarray:
        delta_body = pos_c - ob[0]
        body_dist_sq = jnp.vdot(delta_body, delta_body) - (ob[1] + l) ** 2
        pole_p = get_closest_point(pole, ob[0])
        delta_pole = pole_p - ob[0]
        pole_dist_sq = jnp.vdot(delta_pole, delta_pole) - (ob[1] ** 2)
        return -jnp.array([body_dist_sq, pole_dist_sq])

    return jnp.concatenate([avoid_obs(pos_c, pole, ob) for ob in obs])


@jax.jit
def cost(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.double,
    goal: jnp.ndarray,
    weights: jnp.ndarray,
    Q_T: jnp.ndarray,
) -> jnp.double:
    """Define the problem cost."""
    # Do angle wrapping on theta and phi
    s1_ind = (2, 3)
    state_wrap = get_s1_wrapper(s1_ind)

    delta = state_wrap(x - goal)
    pos_cost = jnp.vdot(delta[:3], delta[:3]) + (1.0 + jnp.cos(x[3]))
    ctrl_cost = jnp.vdot(u - u_hover, u - u_hover)

    stage_cost = weights[0] * pos_cost + weights[1] * ctrl_cost
    term_cost = weights[2] * jnp.vdot(delta, Q_T * delta)

    return jnp.where(t == T, 0.5 * term_cost, 0.5 * stage_cost)


@jax.jit
def state_constraint(
    x: jnp.ndarray,
    t: jnp.double,
    obs: list[tuple[jnp.ndarray, jnp.double]],
    world_range: tuple[jnp.ndarray, jnp.ndarray],
    theta_lim: jnp.double,
) -> jnp.ndarray:
    """Define the state constraints."""
    del t
    theta_cons = jnp.array((x[2] - theta_lim, -x[2] - theta_lim))
    avoid_cons = obs_constraint(x[:4], obs=obs)
    world_cons = jnp.concatenate((-x[:2] + world_range[0], x[:2] - world_range[1]))

    return jnp.concatenate((theta_cons, world_cons, avoid_cons))


@jax.jit
def inequalities(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.double,
    obs: list[tuple[jnp.ndarray, jnp.double]],
    world_range: tuple[jnp.ndarray, jnp.ndarray],
    theta_lim: jnp.double,
    control_bounds: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Define the inequality constraints."""
    control_delta_lb = control_bounds[0] - u
    control_delta_ub = u - control_bounds[1]
    return jnp.concatenate(
        [
            state_constraint(
                x, t, obs=obs, world_range=world_range, theta_lim=theta_lim
            ),
            control_delta_lb,
            control_delta_ub,
        ]
    )


class TestQuadpendulum(unittest.TestCase):
    """Solve a quadpendulum problem."""

    def test(self) -> None:
        """Run the test."""

        key = jax.random.PRNGKey(1234)

        # Confirm mass matrix and inverse computation
        q = jax.random.uniform(key, shape=(4,))
        self.assertTrue(jnp.allclose(get_mass_matrix(q) @ get_mass_inv(q), jnp.eye(4)))  # noqa: PT009

        pos_0 = jnp.array([-2.5, 1.5, 0.0, 0])
        # pos_0 = jnp.array([-3., 0.5, 0., 0])
        pos_g = jnp.array([3.0, -1.5, 0.0, jnp.pi])

        # Solve
        x0 = jnp.concatenate((pos_0, jnp.zeros((4,))))

        weights = jnp.array((0.01, 0.05, 5.0))
        Q_T = jnp.array((10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

        theta_lim = 3.0 * jnp.pi / 4.0

        goal = jnp.concatenate((pos_g, jnp.zeros((4,))))

        obs = [
            (jnp.array([-1.0, 0.5]), 0.5),
            (jnp.array([0.75, -1.0]), 0.75),
            (jnp.array([-2.0, -1.0]), 0.5),
            (jnp.array([2.0, 1.0]), 0.5),
        ]

        world_range = (jnp.array([-4.0, -2.0]), jnp.array([4.0, 2.0]))

        control_bounds = (
            0.1 * Mass * grav * jnp.ones((m,)),
            3.0 * Mass * grav * jnp.ones((m,)),
        )

        c_dim = 0
        g_dim = 2 * m + 2 + 4 + 2 * len(obs)

        X = jnp.tile(x0, (T + 1, 1))
        U = jnp.tile(u_hover, (T, 1))
        S = jnp.zeros([T + 1, g_dim])
        Y_dyn = jnp.zeros_like(X)
        Y_eq = jnp.zeros([T + 1, c_dim])
        Z = jnp.zeros([T + 1, g_dim])

        settings = SolverSettings(
            η0=10.0,
            η_max=1e8,
            η_update_factor=1.1,
            µ0=0.1,
            µ_update_factor=0.9,
            µ_min=1e-9,
            print_logs=True,
        )

        print("Quadpendulum problem")  # noqa: T201
        X, U, S, Y_dyn, Y_eq, Z, iterations, no_errors = solve(
            X_in=X,
            U_in=U,
            S_in=S,
            Y_dyn_in=Y_dyn,
            Y_eq_in=Y_eq,
            Z_in=Z,
            x0=x0,
            cost=partial(cost, goal=goal, weights=weights, Q_T=Q_T),
            dynamics=dynamics,
            inequalities=partial(
                inequalities,
                obs=obs,
                world_range=world_range,
                theta_lim=theta_lim,
                control_bounds=control_bounds,
            ),
            settings=settings,
        )
        self.assertTrue(no_errors)  # noqa: PT009


if __name__ == "__main__":
    jnp.set_printoptions(threshold=1000000)
    jnp.set_printoptions(linewidth=1000000)
    unittest.main()
