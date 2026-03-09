# From https://github.com/google/trajax/blob/main/benchmarks/ilqr_benchmark.py

"""Test goal-reaching with cartpole."""

import unittest
from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsp

import matplotlib.pyplot as plt
from primal_dual_lipa.integrators import euler
from primal_dual_lipa.lagrangian_helpers import pad
from primal_dual_lipa.optimizers import SolverSettings, solve
from primal_dual_lipa.types import Variables
from primal_dual_lipa.vectorization_helpers import vectorize
from tests.helpers import gen_movie, gen_timelapse

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def render_cartpole(ax, state, params, col=None, alpha=1.0):
    """Plots the cartpole on a given axis."""
    _mc, _mp, l = params
    x = state[0]
    theta = state[1]

    # Cart
    cart_w = 0.5
    cart_h = 0.2
    cart_rect = plt.Rectangle(
        (x - cart_w / 2, -cart_h / 2),
        cart_w,
        cart_h,
        color=col if col else "k",
        alpha=alpha,
    )
    ax.add_patch(cart_rect)

    # Pole
    pole_x = x + l * jnp.sin(theta)
    pole_y = -l * jnp.cos(theta)
    ax.plot(
        [x, pole_x],
        [0, pole_y],
        color=col if col else "b",
        alpha=alpha,
        linewidth=2,
    )

    # Joint and Tip
    ax.plot(x, 0, "o", color=col if col else "k", alpha=alpha)
    ax.plot(pole_x, pole_y, "o", color=col if col else "b", alpha=alpha)


@jax.jit
def cartpole(
    state: jax.Array,
    action: jax.Array,
    theta: jax.Array,
    timestep: jnp.double,
    params: jax.Array,
) -> jax.Array:
    """Classic cartpole system.

    Args:
      state: state, (4, ) array
      action: control, (1, ) array
      theta: unused empty global optimization parameters
      timestep: scalar time
      params: tuple of (MASS_CART, MASS_POLE, LENGTH_POLE)

    Returns:
      xdot: state time derivative, (4, )

    """
    del timestep, theta  # Unused

    mc, mp, l = params  # noqa: E741
    g = 9.81

    q = state[0:2]
    qd = state[2:]
    s = jnp.sin(q[1])
    c = jnp.cos(q[1])

    H = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l * l]])
    C = jnp.array([[0.0, -mp * qd[1] * l * s], [0.0, 0.0]])

    G = jnp.array([[0.0], [mp * g * l * s]])
    B = jnp.array([[1.0], [0.0]])

    CqdG = jnp.dot(C, jnp.expand_dims(qd, 1)) + G
    f = jnp.concatenate((qd, jnp.squeeze(-jsp.linalg.solve(H, CqdG, assume_a="pos"))))

    v = jnp.squeeze(jsp.linalg.solve(H, B, assume_a="pos"))
    g = jnp.concatenate((jnp.zeros(2), v))

    return f + g * action


@jax.jit
def goal_cost(
    x: jax.Array,
    u: jax.Array,
    theta: jax.Array,
    t: jnp.int32,
    goal: jnp.int32,
    T: jnp.int32,
) -> jnp.double:
    """Define the cost."""
    del theta  # Unused
    err = x - goal
    stage_cost = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
    final_cost = 1000 * jnp.dot(err, err)
    return jnp.where(t == T, final_cost, stage_cost)


@jax.jit
def goal_equality(
    x: jax.Array,
    _u: jax.Array,
    theta: jax.Array,
    t: jnp.int32,
    goal: jnp.int32,
    T: jnp.int32,
) -> jax.Array:
    """Define the final state constraint."""
    del theta  # Unused
    return jnp.where(t == T, x - goal, jnp.zeros_like(x))


@jax.jit
def inequalities(
    _x: jax.Array,
    u: jax.Array,
    theta: jax.Array,
    t: jnp.int32,
    T: jnp.int32,
) -> jax.Array:
    """Define the control bounds."""
    del theta  # Unused
    return jnp.where(t == T, -jnp.ones(2), jnp.array([u[0] - 5.0, -5.0 - u[0]]))


class TestCartpole(unittest.TestCase):
    """Define the Cartpole test."""

    def test(self) -> None:
        """Run the test."""
        T = 50
        n = 4
        m = 1
        c_dim = n
        g_dim = 2

        x0 = jnp.array([0.0, 0.2, 0.0, -0.1])

        dynamics_params = jnp.array([10.0, 1.0, 0.5])
        dynamics = partial(cartpole, params=dynamics_params)

        dt = 0.1
        dynamics = euler(dynamics, dt=dt)

        goal = jnp.array([0, jnp.pi, 0, 0])
        cost = partial(goal_cost, goal=goal, T=T)

        equalities = partial(goal_equality, goal=goal, T=T)

        inequalities_closure = partial(inequalities, T=T)

        U = jnp.zeros([T, m])
        # X = rollout(dynamics, U, x0)
        X = jnp.linspace(start=x0, stop=goal, num=T + 1)
        S = jnp.zeros([T + 1, g_dim])
        Y_dyn = jnp.zeros_like(X)
        Y_eq = jnp.zeros([T + 1, c_dim])
        Z = jnp.zeros([T + 1, g_dim])

        vars_in = Variables(
            X=X, U=U, S=S, Y_dyn=Y_dyn, Y_eq=Y_eq, Z=Z, Theta=jnp.empty(0)
        )

        # TODO(joao): only change print_logs, if possible.
        settings = SolverSettings(
            η0=10.0,
            η_update_factor=1.1,
            µ0=0.1,
            µ_update_factor=0.95,
            num_iterative_refinement_steps=1,
            print_logs=True,
            # print_ls_logs=True,
        )

        print("Cartpole problem")  # noqa: T201
        vars_out, iterations, no_errors = solve(
            vars_in=vars_in,
            x0=x0,
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities_closure,
            settings=settings,
        )
        self.assertTrue(no_errors)  # noqa: PT009
        self.assertLess(
            vectorize(cost)(
                vars_out.X, pad(vars_out.U), vars_out.Theta, jnp.arange(T + 1)
            ).sum(),
            67.0,
        )  # noqa: PT009

        # Visualization
        print("Generating visualization assets...")  # noqa: T201
        world_range = (jnp.array([-1.0, -0.6]), jnp.array([1.0, 0.6]))
        render_fn = partial(render_cartpole, params=dynamics_params)

        def get_traces(X):
            _mc, _mp, l = dynamics_params
            cart_center = jax.vmap(lambda x: jnp.array([x, 0.0]))(X[:, 0])
            pole_tip = jax.vmap(
                lambda x, theta: jnp.array(
                    [x + l * jnp.sin(theta), -l * jnp.cos(theta)]
                )
            )(X[:, 0], X[:, 1])
            return [cart_center, pole_tip]

        fig, ax = plt.subplots(figsize=(10, 5))
        gen_timelapse(
            ax,
            vars_out.X,
            render_fn,
            world_range,
            5 * 5,
            1.0,
            get_traces_fn=get_traces,
            interpolation_factor=5,
        )
        fig.savefig("cartpole_timelapse.png")
        print("Saved cartpole_timelapse.png")  # noqa: T201

        fig, ax = plt.subplots(figsize=(10, 5))
        anim = gen_movie(
            fig,
            ax,
            vars_out.X,
            render_fn,
            world_range,
            dt,
            get_traces_fn=get_traces,
            interpolation_factor=5,
        )
        anim.save("cartpole_movie.mp4", writer="ffmpeg")
        print("Saved cartpole_movie.mp4")  # noqa: T201


if __name__ == "__main__":
    jnp.set_printoptions(threshold=1000000)
    jnp.set_printoptions(linewidth=1000000)
    unittest.main()
