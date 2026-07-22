"""Defines vectorized helpers for extracting Jacobian and Hessian blocks."""

import jax


def vectorize_edge(fun):  # noqa: ANN001, ANN201
    """Vectorize ``fun(x, u, theta, edge, *args)`` over edges and extra args.

    This maps over ``x``, ``u``, ``edge``, and every additional argument;
    ``theta`` is shared by the batch.
    """

    def vfun(x, u, theta, edge, *args):  # noqa: ANN001, ANN002, ANN202
        # in_axes for (x, u, theta, edge, *args)
        # x: 0, u: 0, theta: None, edge: 0, *args: 0
        in_axes = (0, 0, None, 0) + (0,) * len(args)
        return jax.vmap(fun, in_axes=in_axes)(x, u, theta, edge, *args)

    return vfun


def vectorize_node(fun):  # noqa: ANN001, ANN201
    """Vectorize ``fun(x, theta, node, *args)`` over nodes and extra args."""

    def vfun(x, theta, node, *args):  # noqa: ANN001, ANN002, ANN202
        in_axes = (0, None, 0) + (0,) * len(args)
        return jax.vmap(fun, in_axes=in_axes)(x, theta, node, *args)

    return vfun


def linearize_edge(fun):  # noqa: ANN001, ANN201
    """Vectorized edge Jacobian operator with respect to x, u, and theta."""
    jacobian_x = jax.jacobian(fun, argnums=0)
    jacobian_u = jax.jacobian(fun, argnums=1)
    jacobian_theta = jax.jacobian(fun, argnums=2)

    def linearizer(x, u, theta, edge, *args):  # noqa: ANN001, ANN002, ANN202
        return (
            jacobian_x(x, u, theta, edge, *args),
            jacobian_u(x, u, theta, edge, *args),
            jacobian_theta(x, u, theta, edge, *args),
        )

    return vectorize_edge(linearizer)


def linearize_node(fun):  # noqa: ANN001, ANN201
    """Vectorized Jacobian operator wrt node state and parameters."""
    jacobian_x = jax.jacobian(fun, argnums=0)
    jacobian_theta = jax.jacobian(fun, argnums=1)

    def linearizer(x, theta, node, *args):  # noqa: ANN001, ANN002, ANN202
        return (
            jacobian_x(x, theta, node, *args),
            jacobian_theta(x, theta, node, *args),
        )

    return vectorize_node(linearizer)


def quadratize_edge(fun):  # noqa: ANN001, ANN201
    """Vectorized edge Hessian operator with respect to x, u, and theta."""
    hessian_x = jax.hessian(fun, argnums=0)
    hessian_u = jax.hessian(fun, argnums=1)
    hessian_x_u = jax.jacobian(jax.grad(fun, argnums=0), argnums=1)
    hessian_theta = jax.hessian(fun, argnums=2)
    hessian_x_theta = jax.jacobian(jax.grad(fun, argnums=0), argnums=2)
    hessian_u_theta = jax.jacobian(jax.grad(fun, argnums=1), argnums=2)

    def quadratizer(x, u, theta, edge, *args):  # noqa: ANN001, ANN002, ANN202
        return (
            hessian_x(x, u, theta, edge, *args),
            hessian_u(x, u, theta, edge, *args),
            hessian_x_u(x, u, theta, edge, *args),
            hessian_theta(x, u, theta, edge, *args),
            hessian_x_theta(x, u, theta, edge, *args),
            hessian_u_theta(x, u, theta, edge, *args),
        )

    return vectorize_edge(quadratizer)


def quadratize_node(fun):  # noqa: ANN001, ANN201
    """Vectorized Hessian operator wrt node state and parameters."""
    hessian_x = jax.hessian(fun, argnums=0)
    hessian_theta = jax.hessian(fun, argnums=1)
    hessian_x_theta = jax.jacobian(jax.grad(fun, argnums=0), argnums=1)

    def quadratizer(x, theta, node, *args):  # noqa: ANN001, ANN002, ANN202
        return (
            hessian_x(x, theta, node, *args),
            hessian_theta(x, theta, node, *args),
            hessian_x_theta(x, theta, node, *args),
        )

    return vectorize_node(quadratizer)
