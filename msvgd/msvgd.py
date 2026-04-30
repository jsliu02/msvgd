import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from collections.abc import Iterable
import optax


def listify(val, length):
    '''
    Prepare a numerical/iterable argument for mitosis splits.
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

    @partial(jax.jit, static_argnames=['self', 'h'])
    def svgd_kernel(self, particles, h=-1):
        '''
        Compute the SVGD RBF kernel and its gradient term.
        particles : (k, d)
        returns   : Kxy (k, k), dxkxy (k, d)
        '''
        k = particles.shape[0]
        # Pairwise squared L2 distances  (k, k)
        sq_norms = jnp.sum(particles ** 2, axis=1)          # (k,)
        L2sq = sq_norms[:, None] + sq_norms[None, :] - 2 * particles @ particles.T
        L2sq = jnp.clip(L2sq, 0.0)                          # numerical safety

        h = jnp.where(h <= 0, jnp.median(L2sq) / jnp.log(k), h)

        Kxy = jnp.exp(-L2sq / h)                                # (k, k)
        dxkxy = (Kxy.sum(axis=1, keepdims=True) * particles - Kxy @ particles) * (2.0 / h)    # (k, d)

        return Kxy, dxkxy

    def mitotic_split(self, particles):
        '''
        Double the particle count by concatenating the current particles with a copy.
        In JAX particles are immutable arrays; we return the new array.
        '''
        return jnp.concatenate([particles, particles + 1e-2], axis=0)

    def solve(
        self,
        x0,
        optimizer=optax.adam,
        optimizer_kwargs={"learning_rate": 1e-2},
        max_iter=10_000,
        mitosis_splits=0,
        atol=1e-2,
        rtol=1e-8,
        bandwidth=-1,
        monitor_convergence=0,
    ):
        '''
        Solve mSVGD optimisation.

        Parameters
        ----------
        x0                  : array-like, initial particles (k, d)
        optimizer           : an optax optimizer constructor, or list thereof
        optimizer_kwargs    : dict of kwargs passed to the optimizer, or list thereof
        max_iter            : int or list of ints (one per phase)
        mitosis_splits      : number of particle-doubling steps
        atol, rtol          : convergence tolerances
        bandwidth           : RBF bandwidth (-1 = median heuristic)
        monitor_convergence : int — print max grad every N iterations (0 = disabled)
        '''
        n_phases = mitosis_splits + 1

        optimizer        = listify(optimizer,        n_phases)
        optimizer_kwargs = listify(optimizer_kwargs, n_phases)
        max_iter         = listify(max_iter,         n_phases)
        atol             = listify(atol,             n_phases)
        rtol             = listify(rtol,             n_phases)
        bandwidth        = listify(bandwidth,        n_phases)

        # Initialise particles as a JAX array
        particles = jnp.array(x0)

        for i in range(n_phases):
            k      = particles.shape[0]
            is_MAP = (k == 1)  # static bool — safe to use in Python if

            bw_i   = bandwidth[i]
            atol_i = atol[i]
            rtol_i = rtol[i]
            mc     = monitor_convergence  # static int captured in closure

            opt       = optimizer[i](**optimizer_kwargs[i])
            opt_state = opt.init(particles)

            # ------------------------------------------------------------------
            # Inner step: one gradient + optimizer update.
            # Captured variables (gradient, opt, is_MAP, k, bw_i) are all static
            # from JAX's perspective — they don't change during the while_loop.
            # ------------------------------------------------------------------
            def body_fn(carry):
                particles, opt_state, _, iteration = carry

                # Compute SVGD gradient direction
                grad_particles = -self.gradient(particles)
                if not is_MAP:
                    kxy, dxkxy = self.svgd_kernel(particles, h=bw_i)
                    grad_particles = (kxy @ grad_particles - dxkxy) / k

                # Print max grad every `mc` iterations (no-op when mc == 0)
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

            max_grad = float(jnp.abs(grad_particles).max())
            print(f"Split {i} finished after {int(n_iter)} iterations | max grad = {max_grad:.5f}")

            # ---- mitotic split (except after the last phase) ----
            if i < mitosis_splits:
                particles = self.mitotic_split(particles)

        self.particles = particles
        return particles