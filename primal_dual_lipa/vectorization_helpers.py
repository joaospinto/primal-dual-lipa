# From https://github.com/google/trajax/blob/main/trajax/optimizers.py.
import jax


def vectorize(fun, argnums=3):
    """Returns a jitted and vectorized version of the input function.

    See https://jax.readthedocs.io/en/latest/jax.html#jax.vmap

    Args:
      fun: a numpy function f(*args) to be mapped over.
      argnums: number of leading arguments of fun to vectorize.

    Returns:
      Vectorized/Batched function with arguments corresponding to fun, but extra
      batch dimension in axis 0 for first argnums arguments (x, u, t typically).
      Remaining arguments are not batched.
    """

    def vfun(*args):
        _fun = lambda tup, *margs: fun(*(margs + tup))
        return jax.vmap(_fun, in_axes=(None,) + (0,) * argnums)(
            args[argnums:], *args[:argnums]
        )

    return vfun


def linearize(fun, argnums=3):
    """Vectorized gradient or jacobian operator.

    Args:
      fun: numpy scalar or vector function with signature fun(x, u, t, *args).
      argnums: number of leading arguments of fun to vectorize.

    Returns:
      A function that evaluates Gradients or Jacobians with respect to states and
      controls along a trajectory, e.g.,

          dynamics_jacobians = linearize(dynamics)
          cost_gradients = linearize(cost)
          A, B = dynamics_jacobians(X, pad(U), timesteps)
          q, r = cost_gradients(X, pad(U), timesteps)

          where,
            X is [T+1, n] state trajectory,
            U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
            timesteps is typically np.arange(T+1)

            and A, B are Dynamics Jacobians wrt state (x) and control (u) of
            shape [T+1, n, n] and [T+1, n, m] respectively;

            and q, r are Cost Gradients wrt state (x) and control (u) of
            shape [T+1, n] and [T+1, m] respectively.

            Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jax.jacobian(fun)
    jacobian_u = jax.jacobian(fun, argnums=1)

    def linearizer(*args):
        return jacobian_x(*args), jacobian_u(*args)

    return vectorize(linearizer, argnums)


def quadratize(fun, argnums=3):
    """Vectorized Hessian operator for a scalar function.

    Args:
      fun: numpy scalar with signature fun(x, u, t, *args).
      argnums: number of leading arguments of fun to vectorize.

    Returns:
      A function that evaluates Hessians with respect to state and controls along
      a trajectory, e.g.,

        Q, R, M = quadratize(cost)(X, pad(U), timesteps)

       where,
            X is [T+1, n] state trajectory,
            U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
            timesteps is typically np.arange(T+1)

      and,
            Q is [T+1, n, n] Hessian wrt state: partial^2 fun/ partial^2 x,
            R is [T+1, m, m] Hessian wrt control: partial^2 fun/ partial^2 u,
            M is [T+1, n, m] mixed derivatives: partial^2 fun/partial_x partial_u
    """
    hessian_x = jax.hessian(fun)
    hessian_u = jax.hessian(fun, argnums=1)
    hessian_x_u = jax.jacobian(jax.grad(fun), argnums=1)

    def quadratizer(*args):
        return hessian_x(*args), hessian_u(*args), hessian_x_u(*args)

    return vectorize(quadratizer, argnums)
