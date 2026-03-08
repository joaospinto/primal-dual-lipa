# From https://github.com/google/trajax/blob/main/notebooks/l4dc/QuadPend.ipynb

import unittest
from collections.abc import Callable
from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from matplotlib import animation

from primal_dual_lipa.integrators import euler
from primal_dual_lipa.lagrangian_helpers import pad
from primal_dual_lipa.optimizers import SolverSettings, solve
from primal_dual_lipa.types import Variables
from primal_dual_lipa.vectorization_helpers import vectorize
from tests.helpers import gen_movie, gen_timelapse, get_s1_wrapper

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


n = 8
m = 2
T = 200


Mass = 0.486
mass = 0.2 * Mass
grav = 9.81
l = 0.25  # noqa: E741
L = 2.0 * l
J = 0.00383
fric = 0.01

u_hover = 0.5 * (Mass + mass) * grav * jnp.ones((m,))

r_joint = 0.05 * l
r_tip = 0.1 * l
r_t = 0.3 * l


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
def ode(
    x: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray, t: jnp.double
) -> jnp.ndarray:
    """Provide the dynamics ODE."""
    del theta

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


# Define Quadrotor Geometry (relative coordinates)
quad = (
    jnp.array([[-l, 0.0], [l, 0.0]]),
    jnp.array([[-l, 0.0], [-l, 0.3 * l]]),
    jnp.array([[l, 0.0], [l, 0.3 * l]]),
    jnp.array([[-1.3 * l, 0.3 * l], [-0.7 * l, 0.3 * l]]),
    jnp.array([[0.7 * l, 0.3 * l], [1.3 * l, 0.3 * l]]),
)


def render_quad(ax, state, col=None, alpha=1.0):
    """Plots the quadrotor and its pendulum pole on a given axis."""
    x, y, theta, phi = state[:4]
    pos = jnp.array([x, y])
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

    # Update quad components based on rotation and position
    quad_comps = tuple(v @ R.T + pos for v in quad)

    for comp in quad_comps:
        ax.plot(
            comp[:, 0],
            comp[:, 1],
            color=col if col is not None else "k",
            linewidth=2,
            alpha=alpha,
        )

    # Circumscribing sphere for quad body - should ALWAYS be faint
    pos_c = pos + R @ jnp.array([0.0, 0.15 * l])
    ell = plt.Circle(pos_c, l, alpha=0.05 * alpha, color="k")
    ax.add_patch(ell)

    # Pendulum Pole (line segment only)
    pole_tip = pos + jnp.array([L * jnp.sin(phi), -L * jnp.cos(phi)])
    ax.plot(
        [pos[0], pole_tip[0]],
        [pos[1], pole_tip[1]],
        "-",
        color=col if col is not None else "b",
        alpha=alpha,
    )

    # Pole End Circles
    joint_circ = plt.Circle(
        pos, r_joint, color=col if col is not None else "b", zorder=10, alpha=alpha
    )
    tip_circ = plt.Circle(
        pole_tip, r_tip, color=col if col is not None else "b", zorder=10, alpha=alpha
    )
    ax.add_patch(joint_circ)
    ax.add_patch(tip_circ)


def get_system_geometry(q: jnp.ndarray):
    """Returns the geometry of the quadrotor system."""
    pos = q[:2]
    theta_quad = q[2]
    phi = q[-1]

    R = jnp.array(
        [
            [jnp.cos(theta_quad), -jnp.sin(theta_quad)],
            [jnp.sin(theta_quad), jnp.cos(theta_quad)],
        ]
    )
    pos_c = pos + R @ jnp.array([0.0, 0.15 * l])
    pos_lt = pos + R @ jnp.array([-l, 0.3 * l])
    pos_rt = pos + R @ jnp.array([l, 0.3 * l])

    pole_tip = pos + jnp.array([L * jnp.sin(phi), -L * jnp.cos(phi)])

    # Circles: (center, radius)
    circles = (
        (pos, r_joint),
        (pos_c, l),
        (pos_lt, r_t),
        (pos_rt, r_t),
        (pole_tip, r_tip),
    )
    pole = (pos, pole_tip)
    return circles, pole


def get_closest_point(
    endp: tuple[jnp.ndarray, jnp.ndarray], p_o: jnp.ndarray
) -> jnp.ndarray:
    """Get closest point between point and straight-line between endpoints."""
    x_p, y_p = endp
    t_ = jnp.vdot(p_o - x_p, y_p - x_p) / jnp.vdot(y_p - x_p, y_p - x_p)
    t_min = jnp.minimum(1.0, jnp.maximum(0.0, t_))
    return x_p + t_min * (y_p - x_p)


def obs_constraint(
    q: jnp.ndarray, obs: list[tuple[jnp.ndarray, jnp.double]]
) -> jnp.ndarray:
    """Define the obstacle constraints."""
    circles, pole = get_system_geometry(q)

    def avoid_obs(ob: tuple[jnp.ndarray, jnp.double]) -> jnp.ndarray:
        ob_pos, ob_r = ob
        cons = []
        for c, r in circles:
            delta = c - ob_pos
            cons.append(-(jnp.vdot(delta, delta) - (ob_r + r) ** 2))

        pole_p = get_closest_point(pole, ob_pos)
        delta_pole = pole_p - ob_pos
        cons.append(-(jnp.vdot(delta_pole, delta_pole) - ob_r**2))
        return jnp.array(cons)

    return jnp.concatenate([avoid_obs(ob) for ob in obs])


@jax.jit
def cost(
    x: jnp.ndarray,
    u: jnp.ndarray,
    theta: jnp.ndarray,
    t: jnp.int32,
    goal: jnp.ndarray,
    weights: jnp.ndarray,
    Q_T: jnp.ndarray,
) -> jnp.double:
    """Define the problem cost."""
    del theta
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
def equalities(
    x: jnp.ndarray,
    u: jnp.ndarray,
    theta: jnp.ndarray,
    t: jnp.int32,
    T: jnp.int32,
    goal: jnp.ndarray,
) -> jnp.ndarray:
    """Define the equality constraints."""
    del u, theta
    # State wrapping for the goal comparison
    s1_ind = (2, 3)
    state_wrap = get_s1_wrapper(s1_ind)
    return jnp.where(t == T, state_wrap(x - goal), jnp.zeros_like(x))


@jax.jit
def state_constraint(
    x: jnp.ndarray,
    theta: jnp.ndarray,
    t: jnp.int32,
    obs: list[tuple[jnp.ndarray, jnp.double]],
    world_range: tuple[jnp.ndarray, jnp.ndarray],
    theta_lim: jnp.double,
) -> jnp.ndarray:
    """Define the state constraints."""
    del theta, t
    theta_cons = jnp.array((x[2] - theta_lim, -x[2] - theta_lim))

    circles, _ = get_system_geometry(x[:4])

    # World range constraints for all circles
    world_cons = []
    for c, r in circles:
        world_cons.append(-c + world_range[0] + r)
        world_cons.append(c - world_range[1] + r)
    world_cons = jnp.concatenate(world_cons)

    avoid_cons = obs_constraint(x[:4], obs=obs)

    return jnp.concatenate(
        (
            theta_cons,
            world_cons,
            avoid_cons,
        )
    )


@jax.jit
def inequalities(
    x: jnp.ndarray,
    u: jnp.ndarray,
    theta: jnp.ndarray,
    t: jnp.int32,
    T: jnp.int32,
    obs: list[tuple[jnp.ndarray, jnp.double]],
    world_range: tuple[jnp.ndarray, jnp.ndarray],
    theta_lim: jnp.double,
    control_bounds: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Define the inequality constraints."""
    control_delta_lb = jnp.where(t == T, -jnp.ones_like(u), control_bounds[0] - u)
    control_delta_ub = jnp.where(t == T, -jnp.ones_like(u), u - control_bounds[1])
    return jnp.concatenate(
        [
            state_constraint(
                x, theta, t, obs=obs, world_range=world_range, theta_lim=theta_lim
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

        cost_closure = partial(cost, goal=goal, weights=weights, Q_T=Q_T)

        equalities_closure = partial(equalities, T=T, goal=goal)

        inequalities_closure = partial(
            inequalities,
            T=T,
            obs=obs,
            world_range=world_range,
            theta_lim=theta_lim,
            control_bounds=control_bounds,
        )

        c_dim = equalities_closure(jnp.zeros(n), jnp.zeros(m), jnp.empty(0), 0).size
        g_dim = inequalities_closure(jnp.zeros(n), jnp.zeros(m), jnp.empty(0), 0).size

        X = jnp.tile(x0, (T + 1, 1))
        U = jnp.tile(u_hover, (T, 1))
        S = jnp.zeros((T + 1, g_dim))
        Y_dyn = jnp.zeros_like(X)
        Y_eq = jnp.zeros([T + 1, c_dim])
        Z = jnp.zeros([T + 1, g_dim])

        vars_in = Variables(
            X=X, U=U, S=S, Y_dyn=Y_dyn, Y_eq=Y_eq, Z=Z, Theta=jnp.empty(0)
        )

        # TODO(joao): only change print_logs, if possible.
        settings = SolverSettings(
            residual_sq_threshold=1e-10,
            α_min=0.5,
            η0=10.0,
            η_max=1e9,
            η_update_factor=1.1,
            µ0=0.1,
            µ_update_factor=0.9,
            µ_min=1e-16,
            num_iterative_refinement_steps=1,
            print_logs=True,
            # print_ls_logs=True,
        )

        print("Quadpendulum problem")  # noqa: T201
        vars_out, iterations, no_errors = solve(
            vars_in=vars_in,
            x0=x0,
            cost=cost_closure,
            dynamics=dynamics,
            equalities=equalities_closure,
            inequalities=inequalities_closure,
            settings=settings,
        )
        self.assertTrue(no_errors)  # noqa: PT009
        self.assertLess(
            vectorize(cost_closure)(
                vars_out.X, pad(vars_out.U), vars_out.Theta, jnp.arange(T + 1)
            ).sum(),
            9.4,
        )  # noqa: PT009

        # Visualization
        print("Generating visualization assets...")  # noqa: T201

        def get_traces(X):
            quad_center = X[:, :2]
            pole_tip = jax.vmap(
                lambda x, y, phi: jnp.array(
                    [x + L * jnp.sin(phi), y - L * jnp.cos(phi)]
                )
            )(X[:, 0], X[:, 1], X[:, 3])
            return [quad_center, pole_tip]

        fig, ax = plt.subplots(figsize=(10, 5))
        gen_timelapse(
            ax,
            vars_out.X,
            render_quad,
            world_range,
            10,
            1.0,
            obs=obs,
            get_traces_fn=get_traces,
        )
        fig.savefig("quadpend_timelapse.png")
        print("Saved quadpend_timelapse.png")  # noqa: T201

        fig, ax = plt.subplots(figsize=(10, 5))
        anim = gen_movie(
            fig,
            ax,
            vars_out.X,
            render_quad,
            world_range,
            dt,
            obs=obs,
            get_traces_fn=get_traces,
        )
        anim.save("quadpend_movie.mp4", writer="ffmpeg")
        print("Saved quadpend_movie.mp4")  # noqa: T201


if __name__ == "__main__":
    jnp.set_printoptions(threshold=1000000)
    jnp.set_printoptions(linewidth=1000000)
    unittest.main()
