import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import numpy as np

from functools import partial
from collections.abc import Iterable
import inspect

def _listify(val, length, dtype=None):
    """Broadcast a hyperparameter to one value per phase. Helper -- not user-facing."""
    if isinstance(val, Iterable) and not isinstance(val, (dict, str)):
        if len(val) != length: raise ValueError(
            "Incorrect gradient descent hyperparameter argument length, "
            f"got {len(val)}, expecting {length}.")
        listed = list(val)
    else:
        listed = [val] * length
    return jnp.array(listed, dtype=dtype) if dtype is not None else listed


def _normalize_schedule(k_schedule, k_start):
    """Standardize k_schedule to a strictly increasing list. Helper -- not user-facing."""
    if k_schedule is None: schedule = []
    elif isinstance(k_schedule, Iterable): schedule = list(k_schedule)
    else: schedule = [k_schedule]

    for i, (prev_k, k_target) in enumerate(zip([k_start, *schedule], schedule)):
        if k_target <= prev_k: raise ValueError(
            "k_schedule must be strictly increasing and greater than the starting "
            f"particle count ({k_start}); mSVGD only grows the particle count, it "
            f"doesn't shrink it. Got k_schedule[{i}]={k_target}, expected > {prev_k}.")
    return schedule



class MSVGD():
    def __init__(self, logdensity, data=None):
        '''
        Define the target log-density, up to an additive constant.

        logdensity : (d,) -> scalar. Add a second `data_batch` (n_batch, d_data) argument to
            enable batched gradient descent.
        data       : (n_data, d_data), only needed for batching -- data hard-coded into
            logdensity requires no argument here.
        '''
        self.data = data
        self.particles = None

        # Handle logdensity signature
        n_args = len(inspect.signature(logdensity).parameters)
        if n_args not in (1, 2):
            raise ValueError("The logdensity has an invalid number of arguments (1 by default or 2 if data batching).")
        self._batch_ready = n_args == 2
        self.logdensity = logdensity if self._batch_ready else lambda x, data_batch: logdensity(x)

        def _single_grad(x, data_batch):
            return jax.grad(lambda x: self.logdensity(x, data_batch).sum())(x)
        self.gradient = jax.jit(jax.vmap(_single_grad, in_axes=(0, None)))
        
    @staticmethod
    def _stein_R(particles, raw_grad):
        '''
        Dispersion diagnostic from Stein's identity. With f(x) = x, the identity gives
        E_p[(x - mu) s(x)^T] = -I for s = grad log p, so

            R = -(1/(k*dim)) sum_i (x_i - xbar) . s(x_i)  -->  1  under the target.

        For a Gaussian target and an ensemble of covariance A, R = tr(A Sigma^-1)/dim, i.e. the
        mean variance-inflation factor: 1 is correctly dispersed, < 1 underdispersed. Unlike max
        grad, R's target value is fixed by theory rather than by the problem, so it flags an
        ensemble that has converged to the wrong spread -- SVGD's characteristic failure, and
        one an optimizer's own convergence test cannot see.

        Necessary, not sufficient: a single moment condition, exact only for Gaussian targets,
        and blind to WHICH directions are deficient. Averaging over dim also makes it dominated
        by whichever block holds most of the coordinates.

        raw_grad : (k, dim) -- raw_grad = -self.gradient(particles, data_batch), i.e. -s
        returns  : scalar
        '''
        return jnp.sum((particles - particles.mean(axis=0)) * raw_grad) / particles.size
        
    def pairwise_distance(self, particles, h=-1):
        k = particles.shape[0]
        sq_norms = jnp.sum(particles ** 2, axis=1) # (k,)
        # "highest" precision avoids catastrophic cancellation in this sum-of-squares expansion
        with jax.default_matmul_precision("highest"):
            L2sq = sq_norms[:, None] + sq_norms[None, :] - 2 * particles @ particles.T # (k, k)

        # Adaptive RBF bandwidth: h = median(||x - y||^2), the plain median heuristic. np (not
        # jnp) since k is static: a host-side constant baked into the trace once, rather than an
        # index construction re-run on every call.
        #
        # This used to be median / log(k), following Liu & Wang (NeurIPS 2016). The log(k) is a
        # measurable mistake in high dimensions. Ba, Erdogdu, Ghassemi, Sun, Suzuki, Wu & Zhang,
        # "Understanding the Variance Collapse of SVGD in High Dimensions" (ICLR 2022, OpenReview
        # Qycd9j5Qp9J) prove, for an isotropic Gaussian target in the proportional limit with
        # gamma = d/n > 1, that the plain heuristic gives
        #
        #     Corollary 4:   Var_SVGD / Var_target = (e - 1)^-1 * n / d = 0.582 n/d,
        #
        # whereas the log(k) variant gives ln(k)/d -- measured, not proved, but reproduced to
        # within 1% against Ba et al.'s own law in the same harness over d = 50..608 and
        # k = 10..3200. The ratio is 0.582 k / ln k: a factor of 39 at k = 400 and 281 at k = 4000,
        # always in favour of dropping the log.
        #
        # On real posteriors the gain is smaller, because they are anisotropic and neither law
        # applies exactly: measured on three MAGI benchmarks at d = 306..608, k = 400, dropping
        # the log improves the Mahalanobis energy distance by 1.8-2.4x and the Stein-identity ratio
        # by 5-9x. It is a strict improvement in all six cases tested and costs nothing, but it is
        # NOT a fix -- both variants leave the ensemble 34-48x its Monte-Carlo floor at these
        # dimensions. The bandwidth that matters is a large FIXED one; see the `bandwidth`
        # argument to solve().
        upper_tri = np.triu_indices(k, k=1) # upper triangle, excluding diagonal
        median = jnp.median(jnp.clip(L2sq[upper_tri], min=jnp.array(1e-6, dtype=particles.dtype)))
        return L2sq, jnp.where(h <= 0, median, h) # (1,)

    @partial(jax.jit, static_argnames=['self'])
    def _svgd_update(self, particles, raw_grad, h=-1):
        '''
        Standard SVGD drift-minus-repulsion update, from the joint RBF kernel.

        raw_grad : (k, dim) -- raw_grad = -self.gradient(particles, data_batch)
        returns  : (k, dim) combined update, fed to a descent optimizer
        '''
        L2sq, h = self.pairwise_distance(particles, h)

        k = particles.shape[0]
        Kxy = jnp.exp(-L2sq / h)
        dxkxy = (Kxy.sum(axis=1, keepdims=True) * particles - Kxy @ particles) * (2.0 / h) # (k, dim)
        return (Kxy @ raw_grad - dxkxy) / k

    @partial(jax.jit, static_argnames=['self', 'is_MAP', 'k_target'])
    def _mitotic_split(self, particles, key, is_MAP, k_target):
        '''
        Expand the particle count directly to k_target in one step, using covariance-matched
        jitter: offspring are drawn from a multivariate Gaussian fit to the ensemble's own
        empirical covariance (a smoothed-bootstrap-style perturbation). Every particle is
        anchored to the same number of offspring (n_new // k); only the remainder (n_new % k)
        goes to a random subset of parents, drawn without replacement so no parent gets more
        than one "extra".

        Jitter scale is calibrated to budget = h/2, h being the kernel's median-heuristic
        bandwidth. This matches the kernel's implicit Gaussian variance: the kernel is
        exp(-d^2/h), so h = 2*sigma^2 in the usual Gaussian-exponent convention.
        '''
        k, dim = particles.shape
        n_new = k_target - k
        budget = (0.01 if is_MAP else self.pairwise_distance(particles, -1)[1]) / 2

        key_parents, key_jitter = jr.split(key)
        centered = particles - particles.mean(axis=0)
        cov = (centered.T @ centered) / k + 1e-6 * jnp.eye(dim, dtype=particles.dtype)
        L = jnp.linalg.cholesky(cov)

        n_each, n_remainder = divmod(n_new, k)
        idx = jnp.concatenate([jnp.repeat(jnp.arange(k), n_each),
                               jr.choice(key_parents, k, shape=(n_remainder,), replace=False)])
        z = jr.normal(key_jitter, shape=(n_new, dim), dtype=particles.dtype)
        offspring = particles[idx] + jnp.sqrt(budget / jnp.trace(cov)) * (z @ L.T)

        return jnp.concatenate([particles, offspring], axis=0)

    @partial(jax.jit, static_argnames=[
        'self', 'optimizer', 'opt_kwargs_keys', 'is_MAP', 'batch_size',
        'grad_clip_enabled', 'monitor_enabled'])
    def _run_phase(
        self, particles, data, key, *,
        opt_kwargs_values, grad_clip_value, atol, rtol, bandwidth, max_iter, phase, monitor_interval,
        optimizer, opt_kwargs_keys, is_MAP, batch_size, grad_clip_enabled, monitor_enabled,
    ):
        '''
        Run gradient descent (with optional SVGD kernel / batching) to convergence or max_iter,
        for one mitosis phase.

        JIT-compiled as a single unit, keyed on `self`, the static arguments above, and the
        shapes/dtypes of the array arguments. Hyperparameter *values* stay traced rather than
        baked in as constants, so re-calling `solve()` at the same shapes and static config
        reuses the compiled executable instead of retracing. `phase` only feeds a debug-print
        label and `monitor_interval` a traced modulo, so neither need be static -- only
        `monitor_enabled` changes which code path is compiled in.
        '''
        opt = optimizer(**dict(zip(opt_kwargs_keys, opt_kwargs_values)))
        if grad_clip_enabled:
            opt = optax.chain(optax.clip_by_global_norm(grad_clip_value), opt)
        # vmap init/update over the particle axis rather than calling them once on the whole
        # (k, dim) array. Free, and a no-op for most optimizers, but it matters for any whose
        # step scale is a single scalar pooled over the parameter pytree. It also makes
        # grad_clip a per-particle norm instead of one pooled across all k particles, which is
        # arguably more correct for an ensemble anyway
        opt_state = jax.vmap(opt.init)(particles)

        if batch_size is not None:
            N = data.shape[0]
            n_batches = N // batch_size

        def body_fn(carry):
            particles, opt_state, _, _, iteration, key, data_shuffled = carry
            key, subkey = jr.split(key)

            if batch_size is not None:
                batch_start = (iteration % n_batches) * batch_size
                # Reshuffle each epoch, i.e. whenever the batch index resets to 0 (including the first)
                data_shuffled = jax.lax.cond(batch_start == 0,
                                             lambda: data[jr.permutation(subkey, N)], lambda: data_shuffled)
                data_batch = jax.lax.dynamic_slice_in_dim(data_shuffled, batch_start, batch_size, axis=0)
            else:
                data_batch = data

            grad_particles = -self.gradient(particles, data_batch)
            # read off the raw score, before any kernel rescales it
            stein_R = self._stein_R(particles, grad_particles)

            if not is_MAP:
                grad_particles = self._svgd_update(particles, grad_particles, h=bandwidth)

            if monitor_enabled:
                jax.lax.cond(
                    iteration % monitor_interval == 0,
                    lambda: jax.debug.print("  Split {i} | Iter {it} | Max grad = {m:.5f} | Stein R = {r:.4f}",
                                            i=phase, it=iteration, m=jnp.abs(grad_particles).max(), r=stein_R),
                    lambda: None,
                )

            updates, opt_state = jax.vmap(opt.update)(grad_particles, opt_state, particles)
            particles = optax.apply_updates(particles, updates)
            return (particles, opt_state, grad_particles, stein_R, iteration + 1, key, data_shuffled)

        def cond_fn(carry):
            particles, _, grad_particles, _, iteration, _, _ = carry
            converged = jnp.all(jnp.abs(grad_particles) <= atol + rtol * jnp.abs(particles))
            return ~converged & (iteration < max_iter)

        # Seed grad with inf so the convergence check always runs at least one step
        init_carry = (particles, opt_state, jnp.full_like(particles, jnp.inf),
                      jnp.zeros((), particles.dtype),
                      jnp.zeros((), jnp.int32), key, data if batch_size is not None else None)
        particles, _, grad_particles, stein_R, n_iter, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return particles, grad_particles, stein_R, n_iter

    def solve(
        self,
        x0,
        k_schedule=None,
        random_seed=8,
        data=None,
        monitor_convergence=0,
        optimizer=optax.adam,
        optimizer_kwargs={"learning_rate": 0.1},
        batch_size=None,
        is_MAP=False,
        max_iter=10_000,
        atol=1e-2,
        rtol=1e-8,
        bandwidth=-1,
        grad_clip=None
    ):
        '''
        Solve mSVGD optimization.

        Arguments
        ----------
        x0                  : array-like, initial particles (k, d)
        k_schedule          : int, list of ints, or None (default). None runs one phase at
            len(x0) particles with no growth. Otherwise each entry is the particle count after
            one covariance-matched split, giving len(k_schedule) splits and len(k_schedule)+1
            phases; entries must strictly increase, the first exceeding len(x0). An int is
            shorthand for a single split.
        random_seed         : int used to set jax.random key for sampling the mitotic splits
        data                : override data stored at class initialization
        monitor_convergence : int — print max grad and the Stein-identity dispersion diagnostic
            (see _stein_R; -> 1 under the target, < 1 underdispersed) every N iterations
            (0 = print status after each phase, < 0 = fully silence). The diagnostic is also
            left on self.stein_R regardless.

        ----------
        Note: each argument below takes either one value used for every phase, or a list of
            n_phases values (one per phase). A list of the wrong length is an error.

        optimizer           : an optax optimizer constructor, configured for descent
        optimizer_kwargs    : dict of kwargs passed to the optimizer
            Warning : some optimizer kwargs must share x0's dtype,
                e.g. {"learning_rate" : jnp.array(0.1, dtype=x0.dtype)}
        batch_size          : int for batched optimization, None for the full dataset
        is_MAP              : bool, mode-find on the logdensity gradient alone (no SVGD kernel)
        max_iter            : int, iteration cap for the phase
        atol, rtol          : convergence tolerances,  all(grad <= atol + rtol * particles)
        bandwidth           : RBF bandwidth (-1 = median heuristic)
        grad_clip           : float, max global norm for the particle gradient before the
            optimizer step, None to disable. Guards against exploding updates in
            batched/stochastic optimization.
        '''
        # dtype carries over if x0 was already a JAX array
        particles = jnp.array(x0)

        key = jr.key(random_seed) if isinstance(random_seed, int) else random_seed
        k_schedule = _normalize_schedule(k_schedule, particles.shape[0])
        n_phases = len(k_schedule) + 1

        optimizer         = _listify(optimizer, n_phases)
        optimizer_kwargs  = _listify(optimizer_kwargs, n_phases)
        batch_size        = _listify(batch_size, n_phases)  # None means full batch
        is_MAP            = _listify(is_MAP, n_phases)
        max_iter          = _listify(max_iter, n_phases)
        atol              = _listify(atol, n_phases, particles.dtype)
        rtol              = _listify(rtol, n_phases, particles.dtype)
        bandwidth         = _listify(bandwidth, n_phases, particles.dtype)
        grad_clip         = _listify(grad_clip, n_phases)

        monitor_enabled = monitor_convergence > 0
        # inert when monitoring is off (that path isn't compiled in); 1 avoids a mod-by-zero trace
        monitor_interval = monitor_convergence if monitor_enabled else 1

        if data is not None:
            self.data = data
        if any(b is not None for b in batch_size):
            if self.data is None:
                raise ValueError("Batch size set but no data provided.")
            if not self._batch_ready:
                raise ValueError("Batch size set but logdensity signature does not take data.")
            N = self.data.shape[0]
            batch_size = [b if b is not None and 0 < b < N else None for b in batch_size]

        for i in range(n_phases):
            is_MAP_i = bool(is_MAP[i]) or particles.shape[0] == 1  # a lone particle has no kernel
            clip_on = grad_clip[i] is not None
            opt_keys = tuple(sorted(optimizer_kwargs[i]))
            key_sgd, key_mitosis = jr.split(jr.fold_in(key, i))

            particles, grad_particles, stein_R, n_iter = self._run_phase(
                particles, self.data, key_sgd,
                opt_kwargs_values=tuple(optimizer_kwargs[i][kw] for kw in opt_keys),
                grad_clip_value=grad_clip[i] if clip_on else 0.0,
                atol=atol[i], rtol=rtol[i], bandwidth=bandwidth[i], max_iter=max_iter[i],
                phase=i, monitor_interval=monitor_interval,
                optimizer=optimizer[i], opt_kwargs_keys=opt_keys, is_MAP=is_MAP_i,
                batch_size=batch_size[i], grad_clip_enabled=clip_on,
                monitor_enabled=monitor_enabled
            )

            if monitor_convergence >= 0:
                max_grad = float(jnp.abs(grad_particles).max())
                print(f"Split {i} finished after {int(n_iter)} iterations | "
                      f"max grad = {max_grad:.5f} | Stein R = {float(stein_R):.4f}")

            # Direct-jump split after every phase but the last
            if i < len(k_schedule):
                particles = self._mitotic_split(particles, key_mitosis, is_MAP_i, k_schedule[i])

        self.particles = particles.copy()
        self.stein_R = float(stein_R)
        return self.particles
