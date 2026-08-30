import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import numpy as np

from functools import partial
from collections.abc import Iterable
import inspect

def _listify(val, length, dtype=None):
    """
    Prepare a numerical/iterable argument for mitosis splits.

    Helper function -- not user-facing.
    """
    if isinstance(val, Iterable) and not isinstance(val, (dict, str)):
        if len(val) != length: raise ValueError(
            "Incorrect gradient descent hyperparameter argument length, "
            f"got {len(val)}, expecting {length}.")
        else: listed = val
    else: listed = [val] * length
    return jnp.array(listed, dtype=dtype) if dtype is not None else listed



class MSVGD():
    def __init__(self, logdensity, data=None):
        '''
        Define log-density of the target distribution, may be up to additive constant.

        logdensity: (d,) -> (1,)
        - for batched gradient descent, optionally include a `data_batch` (n_batch, d_data) argument to the logdensity funtion

        data (n_data, d_data): optional data argument for batched optimization. Not needed if all data is hard-coded into logdensity
        '''
        self.data = data

        # Handle logdensity signature
        if len(inspect.signature(logdensity).parameters) == 1:
            self.logdensity = lambda x, y: logdensity(x)
            self._batch_ready = False
        elif len(inspect.signature(logdensity).parameters) == 2:
            self.logdensity = logdensity
            self._batch_ready = True
        else:
            raise ValueError("The logdensity has an invalid number of arguments (1 by default or 2 if data batching).")

        def _single_grad(x, data_batch):
            return jax.grad(lambda x: self.logdensity(x, data_batch).sum())(x)
        self.gradient = jax.jit(jax.vmap(_single_grad, in_axes=(0, None)))

        self.particles = None

    
    def pairwise_distance(self, particles, h=-1):
        k = particles.shape[0]
        # Pairwise squared L2 distances  (k, k)
        sq_norms = jnp.sum(particles ** 2, axis=1) # (k,)
        # "highest" precision avoids catastrophic cancellation in this sum-of-squares expansion
        with jax.default_matmul_precision("highest"):
            L2sq = sq_norms[:, None] + sq_norms[None, :] - 2 * particles @ particles.T

        # Adaptive RBF bandwidth
        log_k = jnp.log(jnp.array(k, dtype=particles.dtype))
        # np (not jnp): k is static, so this is a host-side constant baked into the trace once,
        # Instead of a scatter/reduce-window index construction re-run on every call
        upper_tri = np.triu_indices(k, k=1) # keep upper triangle, excluding diagonal
        h = jnp.where(h <= 0, jnp.quantile(jnp.clip(L2sq[upper_tri], min=jnp.array(1e-6, dtype=particles.dtype)), 0.5) / log_k, h) # (1,)
        return L2sq, h

        
    @partial(jax.jit, static_argnames=['self'])
    def _svgd_kernel(self, particles, h=-1):
        '''
        Compute the SVGD RBF kernel and its gradient term.
        particles : (k, d)
        returns   : Kxy (k, k), dxkxy (k, d)
        '''
        L2sq, h = self.pairwise_distance(particles, h)
        
        Kxy = jnp.exp(-L2sq / h) # (k, k)
        dxkxy = (Kxy.sum(axis=1, keepdims=True) * particles - Kxy @ particles) * (jnp.array(2.0, dtype=particles.dtype) / h) # (k, d)

        return Kxy, dxkxy

        
    @partial(jax.jit, static_argnames=['self', 'is_MAP', 'k_final'])
    def _mitotic_split(self, particles, key, is_MAP, k_final):
        '''
        Expand the particle count from its current size directly to k_final in a single step,
        using covariance-matched jitter: new offspring are drawn from a multivariate Gaussian
        fit to the current ensemble's own empirical covariance (a smoothed-bootstrap-style
        perturbation), each anchored to a randomly-chosen existing particle (sampled uniformly
        with replacement, so k_final need not be an exact multiple of the current particle
        count). Jitter scale is calibrated to budget = h/2, where h is the SVGD kernel's
        median-heuristic bandwidth -- this matches the kernel's own implicit Gaussian variance
        (the kernel is exp(-d^2/h), so h = 2*sigma^2 in the usual Gaussian-exponent
        convention).

        This specific method (covariance-matched jitter, single direct jump rather than
        several intermediate doublings) was chosen after comparing 6 particle-splitting
        strategies across multiple compute budgets, using less compute by skipping intermediate phases
        entirely -- a reasonable tradeoff for a simpler, two-phase solve() API.
        '''
        k, dim = particles.shape
        n_new = k_final - k

        if not is_MAP:
            _, h = self.pairwise_distance(particles, -1)
            budget = h / 2
        else:
            budget = 0.01 / 2

        key1, key2 = jr.split(key)
        mean = jnp.mean(particles, axis=0, keepdims=True)
        centered = particles - mean
        cov = (centered.T @ centered) / k
        cov = cov + 1e-6 * jnp.eye(dim, dtype=particles.dtype)
        L = jnp.linalg.cholesky(cov)

        idx = jr.randint(key1, shape=(n_new,), minval=0, maxval=k)
        source_particles = particles[idx]
        z = jr.normal(key2, shape=(n_new, dim), dtype=particles.dtype)
        scale = jnp.sqrt(budget / jnp.trace(cov))
        offspring = source_particles + scale * (z @ L.T)

        return jnp.concatenate([particles, offspring], axis=0)

    @partial(jax.jit, static_argnames=[
        'self', 'optimizer', 'opt_kwargs_keys', 'is_MAP', 'batch_size',
        'grad_clip_enabled', 'monitor_enabled',
    ])
    def _run_phase(
        self, particles, data, key,
        opt_kwargs_values, grad_clip_value, atol, rtol, bandwidth, max_iter, phase, monitor_interval, *,
        optimizer, opt_kwargs_keys, is_MAP, batch_size, grad_clip_enabled, monitor_enabled,
    ):
        '''
        Run gradient descent (with optional SVGD kernel / batching) to convergence or max_iter,
        for one mitosis phase.

        This is JIT-compiled as a single unit, keyed on `self` plus the static arguments above
        and the shapes/dtypes of the array arguments. Hyperparameter *values* (learning rate,
        tolerances, bandwidth, max_iter, grad-clip norm, phase index, monitor interval) are
        passed as traced arguments rather than baked in as Python constants, so calling `solve()`
        again with the same particle/data shapes and the same static config reuses the compiled
        executable instead of retracing. `phase` only feeds a jax.debug.print label and
        `monitor_interval` only feeds a traced modulo check, so neither needs to be static --
        only whether monitoring is enabled at all (`monitor_enabled`) changes which code path
        gets compiled in. For an example of this use-case, see the annealing in the ring mixure
        example in tests.ipynb.
        '''
        k = particles.shape[0]

        opt = optimizer(**dict(zip(opt_kwargs_keys, opt_kwargs_values)))
        if grad_clip_enabled:
            opt = optax.chain(optax.clip_by_global_norm(grad_clip_value), opt)
        # vmap the optimizer's init/update over the particle axis instead of calling it once
        # on the whole (k, dim) array. This matters for learning-rate-free optimizers in the
        # optax.contrib D-Adaptation family (prodigy, dadapt_adamw): their auto-tuned step
        # scale (e.g. prodigy's `estim_lr`) is a single scalar computed from a norm pooled
        # over every leaf of the parameter pytree -- calling opt.init/update on the (k, dim)
        # array directly means every particle shares ONE scale calibrated from a norm over
        # all k particles combined, so changing k perturbs that calibration and produces a
        # non-monotonic, confounded relationship between particle count and approximation
        # quality. This is a no-op for Adam-style optimizers (their per-coordinate moments
        # are already independent per array element; it does change grad_clip's semantics 
        # from a norm pooled across all k particles to a per-particle norm, which is arguably
        # more correct for a particle ensemble anyway
        opt_state = jax.vmap(opt.init)(particles)

        if batch_size is not None:
            N = data.shape[0]
            n_batches = N // batch_size

        def body_fn(carry):
            particles, opt_state, _, iteration, key, data_shuffled = carry
            key, subkey = jr.split(key)

            # Data batching logic
            if batch_size is not None:
                batch_start = (iteration % n_batches) * batch_size

                # Reshuffle at the start of every epoch, at iterations that reset batch index to 0 (including the first)
                data_shuffled = jax.lax.cond(
                    batch_start == 0,
                    lambda: data[jr.permutation(subkey, N)],
                    lambda: data_shuffled,
                )
                data_batch = jax.lax.dynamic_slice_in_dim(
                    data_shuffled, batch_start, batch_size, axis=0)

            else: data_batch = data

            # Logdensity gradient computation
            grad_particles = -self.gradient(particles, data_batch)

            # Compute SVGD gradient direction
            if not is_MAP:
                kxy, dxkxy = self._svgd_kernel(particles, h=bandwidth)
                grad_particles = (kxy @ grad_particles - dxkxy) / k

            # Print max grad every `monitor_interval` iterations (no output when monitoring disabled)
            if monitor_enabled:
                jax.lax.cond(
                    iteration % monitor_interval == 0,
                    lambda: jax.debug.print(
                        "  Split {i} | Iter {it} | Max grad = {m:.5f}",
                        i=phase, it=iteration, m=jnp.abs(grad_particles).max()
                    ),
                    lambda: None,
                )

            updates, opt_state = jax.vmap(opt.update)(grad_particles, opt_state, particles)
            particles = optax.apply_updates(particles, updates)
            return (particles, opt_state, grad_particles, iteration + 1, key, data_shuffled)

        def cond_fn(carry):
            particles, _, grad_particles, iteration, _, _ = carry
            not_converged = ~jnp.all(
                jnp.abs(grad_particles) <= atol + rtol * jnp.abs(particles)
            )
            under_max_iter = iteration < max_iter
            return not_converged & under_max_iter

        # Seed grad with inf so the convergence check always runs at least one step
        init_grad = jnp.full_like(particles, jnp.inf)
        init_data_shuffled = data if batch_size is not None else None
        init_carry = (particles, opt_state, init_grad, jnp.zeros((), jnp.int32), key, init_data_shuffled)

        particles, _, grad_particles, n_iter, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        return particles, grad_particles, n_iter

    def solve(
        self,
        x0,
        k_final=None,
        random_seed=8,
        data=None,
        optimizer=optax.adam,
        optimizer_kwargs={"learning_rate": 0.1},
        batch_size=None,
        is_MAP=False,
        max_iter=10_000,
        atol=1e-2,
        rtol=1e-8,
        bandwidth=-1,
        grad_clip=None,
        monitor_convergence=0
    ):
        '''
        Solve mSVGD optimization.

        Arguments
        ----------
        x0                  : array-like, initial particles (k, d)
        k_final             : int or None (default). If None, no particle-count growth happens
            -- the whole optimization runs at the len(x0) particles given. If set (must be >
            len(x0)), runs one phase at len(x0) particles, then a single covariance-matched
            split directly to k_final particles (see _mitotic_split), then a second phase at
            k_final particles. A single direct jump is simpler and ~1.7x cheaper than growing
            through several smaller doublings, at a modest coverage cost (measured 44% vs. 47%
            joint coverage on a real inference problem) -- see mitotic_split_variants.py for
            the comparison and the other splitting strategies considered.
        random_seed         : int used to set jax.random key for sampling the mitotic split
        data                : override data stored at class initialization

        Note: The following arguments may each be passed as a single value to be used for both
            phases, or as a list of length 2 (one value per phase) if k_final is set -- a
            length-2 list when k_final is None is an error, since there's only one phase.
        optimizer           : an optax optimizer constructor, or list thereof, configured for descent
        optimizer_kwargs    : dict of kwargs passed to the optimizer, or list thereof
            Warning : It is necessary in some case for optimizer kwargs to have the same dtype as x0,
                e.g. {"learning_rate" : jnp.array(0.1, dtype=x0.dtype)}
        batch_size          : int or list of ints (one per phase) for batched optimization, None for full dataset
        is_MAP              : bool or list of bools for whether to mode-find using on the gradient of only the logdensity
        max_iter            : int or list of ints (one per phase)
        atol, rtol          : convergence tolerances,  all(grad <= atol + rtol * particles)
        bandwidth           : RBF bandwidths (-1 = median heuristic)
        grad_clip           : float or list of floats (one per phase), max global norm for the particle
            gradient before the optimizer step, None to disable. Useful to guard against exploding
            updates in batched/stochastic optimization.

        monitor_convergence : int — print max grad every N iterations
            (0 = print status after each phase, < 0 = fully silence)
        '''
        if isinstance(random_seed, int):
            key = jr.key(random_seed)
        else: key = random_seed

        if k_final is not None and k_final <= x0.shape[0]:
            raise ValueError(
                f"k_final ({k_final}) must be greater than the starting particle count "
                f"({x0.shape[0]}); mSVGD only grows the particle count, it doesn't shrink it."
            )

        n_phases = 1 if k_final is None else 2

        optimizer        = _listify(optimizer, n_phases)
        optimizer_kwargs = _listify(optimizer_kwargs, n_phases)
        batch_size       = _listify(batch_size, n_phases)  # None means full batch
        is_MAP           = _listify(is_MAP, n_phases)
        max_iter         = _listify(max_iter, n_phases)
        atol             = _listify(atol, n_phases, x0.dtype)
        rtol             = _listify(rtol, n_phases, x0.dtype)
        bandwidth        = _listify(bandwidth, n_phases, x0.dtype)
        grad_clip        = _listify(grad_clip, n_phases)

        monitor_enabled = monitor_convergence > 0
        # only matters when monitor_enabled is False, in which case the value is inert
        # (the print code path isn't even compiled in) -- 1 avoids a div/mod-by-zero trace
        monitor_interval = monitor_convergence if monitor_enabled else 1

        if data is not None:
            self.data = data
        if any(batch_size):
            if self.data is None:
                raise ValueError("Batch size set but no data provided.")
            if not self._batch_ready:
                raise ValueError("Batch size set but logdensity signature does not take data.")

            N = self.data.shape[0]
            batch_size = [b if b is not None and 0 < b < N else None for b in batch_size]

        # Ensure that particles are a JAX array
        # Typing will carry over if x0 was originally passed as a JAX array
        particles = jnp.array(x0)

        for i in range(n_phases):
            k = particles.shape[0]
            is_MAP_i = bool(is_MAP[i]) or (k == 1)  # No SVGD kernel if doing MAP estimation

            batch_size_i = batch_size[i]

            grad_clip_i = grad_clip[i]
            grad_clip_enabled = grad_clip_i is not None
            grad_clip_value = grad_clip_i if grad_clip_enabled else 0.0

            opt_kwargs_keys = tuple(sorted(optimizer_kwargs[i].keys()))
            opt_kwargs_values = tuple(optimizer_kwargs[i][kw] for kw in opt_kwargs_keys)

            key_sgd, key_mitosis = jr.split(jr.fold_in(key, i))

            particles, grad_particles, n_iter = self._run_phase(
                particles, self.data, key_sgd,
                opt_kwargs_values, grad_clip_value, atol[i], rtol[i], bandwidth[i], max_iter[i],
                i, monitor_interval,
                optimizer=optimizer[i],
                opt_kwargs_keys=opt_kwargs_keys,
                is_MAP=is_MAP_i,
                batch_size=batch_size_i,
                grad_clip_enabled=grad_clip_enabled,
                monitor_enabled=monitor_enabled,
            )

            if monitor_convergence >= 0:
                max_grad = float(jnp.abs(grad_particles).max())
                print(f"Split {i} finished after {int(n_iter)} iterations | max grad = {max_grad:.5f}")

            # Single direct split to k_final, after the first (and only the first) phase
            if i == 0 and k_final is not None:
                particles = self._mitotic_split(particles, key_mitosis, is_MAP_i, k_final)

        self.particles = particles.copy()
        return particles