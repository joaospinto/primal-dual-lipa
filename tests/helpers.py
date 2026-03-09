from collections.abc import Callable

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from matplotlib import animation


def _wrap_to_pi(x: jax.Array) -> jax.Array:
    """Wrap x to lie within [-pi, pi]."""
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def get_s1_wrapper(s1_ind: tuple[int, ...]) -> Callable[[jax.Array], jax.Array]:
    """Return a function for wrapping S1 components of state to [-pi, pi]."""
    idxs = jnp.array(s1_ind)

    def state_wrapper(x: jax.Array) -> jax.Array:
        return x.at[idxs].set(_wrap_to_pi(x[idxs]))

    return jax.jit(state_wrapper)


def interpolate_trajectory(X: jax.Array, factor: int) -> jax.Array:
    """Interpolate trajectory X by a given factor."""
    if factor <= 1:
        return X
    T_orig = X.shape[0]
    T_new = (T_orig - 1) * factor + 1
    t_orig = jnp.linspace(0, 1, T_orig)
    t_new = jnp.linspace(0, 1, T_new)

    def interp_dim(x_dim):
        return jnp.interp(t_new, t_orig, x_dim)

    return jax.vmap(interp_dim, in_axes=1, out_axes=1)(X)


def gen_timelapse(
    ax: plt.Axes,
    X: jax.Array,
    render_fn: Callable,
    world_range: tuple[jax.Array, jax.Array],
    step0: int,
    stepr: float,
    obs: list | None = None,
    get_traces_fn: Callable[[jax.Array], list[jax.Array]] | None = None,
    interpolation_factor: int = 1,
) -> None:
    """Generates a timelapse plot of the trajectory."""
    X_interp = interpolate_trajectory(X, interpolation_factor)

    if obs is not None:
        for ob in obs:
            ax.add_patch(plt.Circle(ob[0], ob[1], color="k", alpha=0.5))

    if get_traces_fn is not None:
        traces = get_traces_fn(X_interp)
        for trace in traces:
            ax.plot(trace[:, 0], trace[:, 1], "k--", alpha=0.3, linewidth=1)

    ax.set_xlim([world_range[0][0], world_range[1][0]])
    ax.set_ylim([world_range[0][1], world_range[1][1]])
    ax.set_aspect("equal")

    tt = 0
    it = 0
    while tt < X_interp.shape[0]:
        col = "g" if tt == 0 else None
        # Snapshot alpha is typically lower
        render_fn(ax, X_interp[tt], col=col, alpha=0.3 if tt > 0 else 1.0)
        tt += int(step0 * (stepr**it))
        it += 1

    # Render final state in red
    render_fn(ax, X_interp[-1], col="r", alpha=1.0)


def gen_movie(
    fig: plt.Figure,
    ax: plt.Axes,
    X: jax.Array,
    render_fn: Callable,
    world_range: tuple[jax.Array, jax.Array],
    dt: float,
    obs: list | None = None,
    get_traces_fn: Callable[[jax.Array], list[jax.Array]] | None = None,
    interpolation_factor: int = 1,
) -> animation.FuncAnimation:
    """Generates an animation of the system following the trajectory."""
    X_interp = interpolate_trajectory(X, interpolation_factor)
    dt_interp = dt / interpolation_factor

    # Pre-calculate traces if provided
    traces = get_traces_fn(X_interp) if get_traces_fn is not None else []

    def render(tt):
        ax.clear()
        render_fn(ax, X_interp[tt])

        if obs is not None:
            for ob in obs:
                ax.add_patch(plt.Circle(ob[0], ob[1], color="k", alpha=0.3))

        # Render a short trailing trace of the trajectory
        tt_s = int(max(tt - round(0.5 / dt_interp), 0))
        for trace in traces:
            ax.plot(trace[tt_s:tt, 0], trace[tt_s:tt, 1], "r-", linewidth=2, alpha=0.5)

        ax.set_xlim([world_range[0][0], world_range[1][0]])
        ax.set_ylim([world_range[0][1], world_range[1][1]])
        ax.set_aspect("equal", adjustable="box")
        return [ax]

    anim = animation.FuncAnimation(
        fig,
        render,
        frames=range(0, X_interp.shape[0]),
        interval=1000 * dt_interp,
        repeat_delay=3000,
    )

    return anim
