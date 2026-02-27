"""Defines vectorized helpers for extracting Jacobian and Hessian blocks."""

import jax


def vectorize(fun):  # noqa: ANN001, ANN201
    """Return a jitted and vectorized version of the input function.

    The function fun is expected to have signature fun(x, u, theta, t, *args).
    This vectorizer maps over x, u, t and any additional positional args in *args.
    theta (index 2) is NOT batched.
    """

    def vfun(x, u, theta, t, *args):  # noqa: ANN001, ANN002, ANN202
        # in_axes for (x, u, theta, t, *args)
        # x: 0, u: 0, theta: None, t: 0, *args: 0
        in_axes = (0, 0, None, 0) + (0,) * len(args)
        return jax.vmap(fun, in_axes=in_axes)(x, u, theta, t, *args)

    return vfun


def linearize(fun):  # noqa: ANN001, ANN201
    """Vectorized gradient or jacobian operator wrt x, u, theta (indices 0, 1, 2)."""
    jacobian_x = jax.jacobian(fun, argnums=0)
    jacobian_u = jax.jacobian(fun, argnums=1)
    jacobian_theta = jax.jacobian(fun, argnums=2)

    def linearizer(x, u, theta, t, *args):  # noqa: ANN001, ANN002, ANN202
        return (
            jacobian_x(x, u, theta, t, *args),
            jacobian_u(x, u, theta, t, *args),
            jacobian_theta(x, u, theta, t, *args),
        )

    return vectorize(linearizer)


def quadratize(fun):  # noqa: ANN001, ANN201
    """Vectorized Hessian operator wrt x, u, theta (indices 0, 1, 2)."""
    hessian_x = jax.hessian(fun, argnums=0)
    hessian_u = jax.hessian(fun, argnums=1)
    hessian_x_u = jax.jacobian(jax.grad(fun, argnums=0), argnums=1)
    hessian_theta = jax.hessian(fun, argnums=2)
    hessian_x_theta = jax.jacobian(jax.grad(fun, argnums=0), argnums=2)
    hessian_u_theta = jax.jacobian(jax.grad(fun, argnums=1), argnums=2)

    def quadratizer(x, u, theta, t, *args):  # noqa: ANN001, ANN002, ANN202
        return (
            hessian_x(x, u, theta, t, *args),
            hessian_u(x, u, theta, t, *args),
            hessian_x_u(x, u, theta, t, *args),
            hessian_theta(x, u, theta, t, *args),
            hessian_x_theta(x, u, theta, t, *args),
            hessian_u_theta(x, u, theta, t, *args),
        )

    return vectorize(quadratizer)
