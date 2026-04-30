import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from functools import partial
from collections.abc import Iterable

def _listify(val, length):
    '''
    Helper function to prepare a numerical/iterable argument for mitosis splits.
    Not user-facing.
    '''
    if isinstance(val, Iterable) and type(val) is not dict:
        if len(val) == length:
            return jnp.array(val)
        else:
            raise ValueError(
                f"Incorrect gradient descent hyperparameter argument length, "
                f"got {len(val)}, expecting {length}."
            )
    else:
        return [val] * length


class MSVGD():
    def __init__(self, logdensity):
        '''
        Define log-density of the target distribution, may be up to additive constant.
        '''
        # grad of log-density w.r.t. a single particle, then vmap over all particles
        _single_grad = jax.grad(lambda x: logdensity(x).sum())
        self.gradient = jax.jit(jax.vmap(_single_grad, in_axes=0))
        self.particles = None

    @partial(jax.jit, static_argnames=['self'])
    def _svgd_kernel(self, particles, h=-1):
        '''
        Compute the SVGD RBF kernel and its gradient term.
        particles : (k, d)
        returns   : Kxy (k, k), dxkxy (k, d)
        '''
        k = particles.shape[0]
        # Pairwise squared L2 distances  (k, k)
        sq_norms = jnp.sum(particles ** 2, axis=1) # (k,)
        L2sq = sq_norms[:, None] + sq_norms[None, :] - 2 * particles @ particles.T
        # adaptive RBF bandwidth
        h = jnp.where(h <= 0, jnp.median(jnp.clip(L2sq, 0.0)) / jnp.log(k), h) # (1,)

        Kxy = jnp.exp(-L2sq / h) # (k, k)
        dxkxy = (Kxy.sum(axis=1, keepdims=True) * particles - Kxy @ particles) * (2.0 / h) # (k, d)

        return Kxy, dxkxy

    @partial(jax.jit, static_argnames=["self"])
    def _mitotic_split(self, particles, key):
        '''
        Double the particle count by concatenating the current particles with a copy.
        In JAX particles are immutable arrays; we return the new array.
        '''
        k = particles.shape[0]
        sq_norms = jnp.sum(particles ** 2, axis=1) # (k,)
        L2sq = sq_norms[:, None] + sq_norms[None, :] - 2 * particles @ particles.T # (k, k)
        h = jnp.median(jnp.clip(L2sq, 0.0)) / jnp.log(k) # (1,)

        jitter = jr.normal(key, shape=particles.shape) * jnp.sqrt(h / 2) # (k, d)
        return jnp.concatenate([particles, particles + jitter], axis=0)

    def solve(
        self,
        x0,
        mitosis_splits=0,
        key=jr.PRNGKey(8),
        optimizer=optax.adam,
        optimizer_kwargs={"learning_rate": 1e-2},
        max_iter=10_000,
        atol=1e-2,
        rtol=1e-8,
        bandwidth=-1,
        monitor_convergence=0,
    ):
        '''
        Solve mSVGD optimization.

        Arguments
        ----------
        x0                  : array-like, initial particles (k, d)
        mitosis_splits      : number of particle-doubling steps
        key                 : a jax.random key to sample mitosis jitters

        Note: The following arguments may each be passed as a single value to be used globally
            or as a list of length `mitosis_splits+1`, containing (different) values for each mitosis phase.
        optimizer           : an optax optimizer constructor, or list thereof
        optimizer_kwargs    : dict of kwargs passed to the optimizer, or list thereof
        max_iter            : int or list of ints (one per phase)
        atol, rtol          : convergence tolerances,  all(grad <= atol + rtol * particles)
        bandwidth           : RBF bandwidths (-1 = median heuristic)

        monitor_convergence : int — print max grad every N iterations
            (0 = print status after each mitosis split, < 0 = fully silence)
        '''
        n_phases = mitosis_splits + 1

        optimizer        = _listify(optimizer,        n_phases)
        optimizer_kwargs = _listify(optimizer_kwargs, n_phases)
        max_iter         = _listify(max_iter,         n_phases)
        atol             = _listify(atol,             n_phases)
        rtol             = _listify(rtol,             n_phases)
        bandwidth        = _listify(bandwidth,        n_phases)

        # ensure that particles are a JAX array
        # typing will carry over if x0 was originally passed as a JAX array
        particles = jnp.array(x0)

        for i in range(n_phases):
            k      = particles.shape[0]
            is_MAP = (k == 1)  # no SVGD kernel if doing MAP estimation

            bw_i   = bandwidth[i]
            atol_i = atol[i]
            rtol_i = rtol[i]
            mc     = monitor_convergence

            opt       = optimizer[i](**optimizer_kwargs[i])
            opt_state = opt.init(particles)

            # ------------------------------------------------------------------
            # Inner step: one gradient + optimizer update.
            # Captured variables (gradient, opt, is_MAP, k, bw_i) are all static
            # From JAX's perspective — they don't change during the while_loop.
            # ------------------------------------------------------------------
            def body_fn(carry):
                particles, opt_state, _, iteration = carry

                # Compute SVGD gradient direction
                grad_particles = -self.gradient(particles)
                if not is_MAP:
                    kxy, dxkxy = self._svgd_kernel(particles, h=bw_i)
                    grad_particles = (kxy @ grad_particles - dxkxy) / k

                # Print max grad every `mc` iterations (no output when mc == 0)
                if mc > 0:
                    jax.lax.cond(
                        iteration % mc == 0,
                        lambda: jax.debug.print(
                            "  Split {i} | Iter {it} | Max grad = {m:.5f}",
                            i=i, it=iteration, m=jnp.abs(grad_particles).max()
                        ),
                        lambda: None,
                    )

                updates, opt_state = opt.update(grad_particles, opt_state, particles)
                particles = optax.apply_updates(particles, updates)
                return (particles, opt_state, grad_particles, iteration + 1)

            def cond_fn(carry):
                particles, _, grad_particles, iteration = carry
                not_converged = ~jnp.all(
                    jnp.abs(grad_particles) <= atol_i + rtol_i * jnp.abs(particles)
                )
                under_max_iter = iteration < max_iter[i]
                return not_converged & under_max_iter

            # Seed grad with inf so the convergence check always runs at least one step
            init_grad = jnp.full_like(particles, jnp.inf)
            init_carry = (particles, opt_state, init_grad, jnp.zeros((), jnp.int32))

            particles, _, grad_particles, n_iter = jax.lax.while_loop(
                cond_fn, body_fn, init_carry
            )

            if mc >= 0:
                max_grad = float(jnp.abs(grad_particles).max())
                print(f"Split {i} finished after {int(n_iter)} iterations | max grad = {max_grad:.5f}")

            # Mitotic split (except after the last phase)
            if i < mitosis_splits:
                particles = self._mitotic_split(particles, jr.fold_in(key, i))

        self.particles = particles
        return particles