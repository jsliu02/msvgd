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

    def pairwise_distance(self, particles, h=-1):
        k = particles.shape[0]
        sq_norms = jnp.sum(particles ** 2, axis=1) # (k,)
        # "highest" precision avoids catastrophic cancellation in this sum-of-squares expansion
        with jax.default_matmul_precision("highest"):
            L2sq = sq_norms[:, None] + sq_norms[None, :] - 2 * particles @ particles.T # (k, k)

        # Adaptive RBF bandwidth. np (not jnp) since k is static: a host-side constant baked
        # into the trace once, rather than an index construction re-run on every call
        upper_tri = np.triu_indices(k, k=1) # upper triangle, excluding diagonal
        median = jnp.median(jnp.clip(L2sq[upper_tri], min=jnp.array(1e-6, dtype=particles.dtype)))
        return L2sq, jnp.where(h <= 0, median / jnp.log(jnp.array(k, dtype=particles.dtype)), h) # (1,)

    @staticmethod
    def _combine(particles, raw_grad, K, h, drift):
        '''
        Assemble a drift-minus-repulsion update from a kernel matrix K and its bandwidth h.

        K       : (k, k) kernel matrix
        drift   : coefficient on the attraction term (see _reweighted_svgd_update)
        returns : (k, dim) combined update, fed to a descent optimizer
        '''
        k = particles.shape[0]
        dxkxy = (K.sum(axis=1, keepdims=True) * particles - K @ particles) * (2.0 / h) # (k, dim)
        return (drift * (K @ raw_grad) - dxkxy) / k

    @partial(jax.jit, static_argnames=['self'])
    def _svgd_update(self, particles, raw_grad, h=-1):
        '''
        Standard SVGD drift-minus-repulsion update, from the joint RBF kernel.

        raw_grad : (k, dim) -- raw_grad = -self.gradient(particles, data_batch)
        returns  : (k, dim) combined update, fed to a descent optimizer
        '''
        L2sq, h = self.pairwise_distance(particles, h)
        return self._combine(particles, raw_grad, jnp.exp(-L2sq / h), h, drift=1.0)

    @partial(jax.jit, static_argnames=['self'])
    def _reweighted_svgd_update(self, particles, raw_grad, data_batch, h=-1, clip_exponent=20.0):
        '''
        Local KL Convergence Rate for Stein Variational Gradient Descent with Reweighted Kernel
        Xunpeng Huang, Hanze Dong, Cong Fang (2023)
        https://openreview.net/forum?id=k2CRIF8tJ7Y

        Drift-minus-repulsion update under the density-reweighted kernel (Eq. 24)
        k(x,y) = p_*(x)^(-1/2) * k_base(x,y) * p_*(y)^(-1/2), k_base being _svgd_update's RBF
        kernel. Reweighting by the target's inverse-sqrt density amplifies repulsion in
        low-density regions, where the standard kernel's corrective gradient vanishes as the
        particle density -> 0 -- the mechanism behind SVGD's variance-collapse/underdispersion.

        The product rule on the p_*(x)^(-1/2) factor, with s(x) = grad log p_*(x), gives

            grad_x k(x,y) = k(x,y) * [grad_x log k_base(x,y) - 0.5*s(x)]
                          = k(x,y) * [-2(x-y)/h - 0.5*s(x)]

        so substituting into phi(x_i) = (1/k) sum_j [k(x_j,x_i)*s(x_j) + grad_{x_j} k(x_j,x_i)]
        leaves the repulsion term as _svgd_update's and exactly halves the drift coefficient --
        hence drift=0.5 below rather than 1.0.

        Two engineering additions, neither from the paper. First, exp(-0.5*logdensity(x)) is
        defined only up to logdensity's arbitrary additive constant and would overflow outright,
        so logdensity is centered by its per-batch max (densest particle gets factor 1, the rest
        amplify, matching the paper's intent) and the pairwise exponent clipped to
        `clip_exponent`. Second, that centering leaves every factor >= 1, routinely by orders of
        magnitude in high dimensions, so the update's magnitude is not comparable to
        _svgd_update's and a shared atol/rtol would be meaningless. The update is therefore
        returned UNCHANGED alongside the reweight matrix's batch mean as `scale`, which the
        caller divides by for monitoring/atol only -- never for the optimizer step, keeping the
        rescaling exactly invisible to the trajectory for any optimizer.

        NOTE: in high dimensions this kernel needs an Adam-style optimizer; failing that,
        aggressive gradient clipping may mitigate the issue.

        raw_grad   : (k, dim) -- raw_grad = -self.gradient(particles, data_batch)
        data_batch : passed to self.logdensity, matching self.gradient's convention
        returns    : ((k, dim) combined update, scalar scale for monitoring/atol only)
        '''
        L2sq, h = self.pairwise_distance(particles, h)

        logdensity = jax.vmap(lambda x: self.logdensity(x, data_batch).sum())(particles) # (k,)
        ld = logdensity - jnp.max(logdensity) # <= 0; 0 at the batch's highest-density particle
        reweight = jnp.exp(jnp.clip(-0.5 * (ld[:, None] + ld[None, :]), max=clip_exponent)) # (k, k), >= 1

        combined = self._combine(particles, raw_grad, reweight * jnp.exp(-L2sq / h), h, drift=0.5)
        return combined, jnp.mean(reweight)

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
        'grad_clip_enabled', 'monitor_enabled', 'reweighted_kernel'])
    def _run_phase(
        self, particles, data, key, *,
        opt_kwargs_values, grad_clip_value, atol, rtol, bandwidth, max_iter, phase, monitor_interval,
        optimizer, opt_kwargs_keys, is_MAP, batch_size, grad_clip_enabled, monitor_enabled, reweighted_kernel,
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
            particles, opt_state, _, iteration, key, data_shuffled = carry
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

            # `monitor_grad` is what atol/rtol check and what gets printed; the reweighted kernel
            # rescales it by its own typical magnitude so one atol/rtol suits either kernel
            if is_MAP:
                monitor_grad = grad_particles
            elif reweighted_kernel:
                grad_particles, scale = self._reweighted_svgd_update(
                    particles, grad_particles, data_batch, h=bandwidth)
                monitor_grad = grad_particles / scale
            else:
                grad_particles = self._svgd_update(particles, grad_particles, h=bandwidth)
                monitor_grad = grad_particles

            if monitor_enabled:
                jax.lax.cond(
                    iteration % monitor_interval == 0,
                    lambda: jax.debug.print("  Split {i} | Iter {it} | Max grad = {m:.5f}",
                                            i=phase, it=iteration, m=jnp.abs(monitor_grad).max()),
                    lambda: None,
                )

            updates, opt_state = jax.vmap(opt.update)(grad_particles, opt_state, particles)
            particles = optax.apply_updates(particles, updates)
            return (particles, opt_state, monitor_grad, iteration + 1, key, data_shuffled)

        def cond_fn(carry):
            particles, _, monitor_grad, iteration, _, _ = carry
            converged = jnp.all(jnp.abs(monitor_grad) <= atol + rtol * jnp.abs(particles))
            return ~converged & (iteration < max_iter)

        # Seed grad with inf so the convergence check always runs at least one step
        init_carry = (particles, opt_state, jnp.full_like(particles, jnp.inf),
                      jnp.zeros((), jnp.int32), key, data if batch_size is not None else None)
        particles, _, monitor_grad, n_iter, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return particles, monitor_grad, n_iter

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
        grad_clip=None,
        reweighted_kernel=False,
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
        monitor_convergence : int — print max grad every N iterations
            (0 = print status after each phase, < 0 = fully silence)

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
        reweighted_kernel   : bool, use the density-reweighted kernel (Huang, Dong, Fang [2023],
            see _reweighted_svgd_update) rather than the standard joint RBF kernel. Amplifies
            repulsion in low-density regions, countering SVGD's variance-collapse; gave the best
            credible-interval calibration of the corrective techniques tried on a real
            high-dimensional ODE-inference benchmark. No effect when is_MAP is True.
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
        reweighted_kernel = _listify(reweighted_kernel, n_phases)

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

            particles, monitor_grad, n_iter = self._run_phase(
                particles, self.data, key_sgd,
                opt_kwargs_values=tuple(optimizer_kwargs[i][kw] for kw in opt_keys),
                grad_clip_value=grad_clip[i] if clip_on else 0.0,
                atol=atol[i], rtol=rtol[i], bandwidth=bandwidth[i], max_iter=max_iter[i],
                phase=i, monitor_interval=monitor_interval,
                optimizer=optimizer[i], opt_kwargs_keys=opt_keys, is_MAP=is_MAP_i,
                batch_size=batch_size[i], grad_clip_enabled=clip_on,
                monitor_enabled=monitor_enabled, reweighted_kernel=bool(reweighted_kernel[i]),
            )

            if monitor_convergence >= 0:
                max_grad = float(jnp.abs(monitor_grad).max())
                print(f"Split {i} finished after {int(n_iter)} iterations | max grad = {max_grad:.5f}")

            # Direct-jump split after every phase but the last
            if i < len(k_schedule):
                particles = self._mitotic_split(particles, key_mitosis, is_MAP_i, k_schedule[i])

        self.particles = particles.copy()
        return self.particles
