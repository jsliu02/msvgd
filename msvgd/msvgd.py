"""
Mitosis-SVGD: Stein variational gradient descent with a growing particle count.

Read this before using it for posterior uncertainty
---------------------------------------------------
SVGD's equilibrium ensemble is UNDERDISPERSED in high dimensions, and the deficit is a property
of the algorithm rather than of the optimisation. Ba, Erdogdu, Ghassemi, Sun, Suzuki, Wu & Zhang,
"Understanding the Variance Collapse of SVGD in High Dimensions" (ICLR 2022, OpenReview
Qycd9j5Qp9J) prove for an isotropic Gaussian target under the plain median heuristic that

    Var_SVGD / Var_target = (e - 1)^-1 K / d = 0.582 K / d       (their Corollary 4)

so at K < d the ensemble is too narrow in proportion to K/d. Started at EXACT draws from the
target it moves away from them, which rules out any initialisation strategy. The three tools
below exist because of that; `min_particles`, `h_star` and `rescale_stein` are all it takes to
avoid the trap, and `stein_R` is how you notice you are in it.

  * The bandwidth is the plain median, NOT median/log(K). See pairwise_distance -- the log
    costs a factor of 0.582 K / ln K, and it changes the particle requirement's growth rate,
    not just its constant. Measured on isotropic Gaussians, particles needed to reach 90% of
    the correct variance:

        dimension      2    3    4    5    6    7    8
        h = Med/lnK   96  192  512 2048 4096    -    -
        h = Med       16   24   32   48   64   64   96

  * `stein_R` reports the Stein-identity dispersion ratio, which tends to 1 under the target and
    below 1 when underdispersed. It is the only cheap diagnostic that sees this failure: the
    optimiser's own convergence test cannot, because the ensemble does converge -- to the wrong
    spread. It is printed alongside max grad and left on self.stein_R.

  * A large FIXED `bandwidth` removes the feedback loop that drives the collapse (the median
    heuristic measures h from the ensemble, so contraction tightens h, which permits more
    contraction). `h_star` gives the natural scale; 10x it works. This needs a small fixed step
    size -- adaptive optimisers such as Prodigy are unstable in that regime.

  * `rescale_stein` applies the one-number correction that a fixed bandwidth leaves behind. With
    the collapse converted from a shape error to a scale error, dividing the deviations from the
    ensemble mean by sqrt(R) corrects it, with no reference and no extra gradient evaluations.

What none of this fixes
-----------------------
On a target that is not Gaussian, SVGD's fixed point is displaced from the posterior mean along
the axis joining the mode to the mean, by a fraction of that axis's length. It vanishes on an
exact Gaussian and it is not removable by bandwidth, metric, step size or particle count at any
budget we could reach. If the quantity you want is a posterior MEAN of a non-Gaussian target,
use something else; if you want a well-dispersed ensemble, the recipe above delivers one.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import numpy as np

from functools import partial
from collections.abc import Iterable
import inspect
import warnings

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
        
    # ------------------------------------------------------------------ sizing and diagnostics
    @staticmethod
    def min_particles(dim, level=0.9):
        """
        Particles needed for the equilibrium ensemble to reach `level` of the correct variance.

        Measured on isotropic N(0, I_dim), Adam at 0.05, 4000 iterations, started at exact draws;
        `investigation8/exp06b_Kcrit_nolog.py` in the companion repository. Fitted over
        dim = 2..8:

            level 0.8:  K = 8.5 exp(0.132 dim)
            level 0.9:  K = 10.0 exp(0.287 dim)

        reproducing the measured critical counts to within one rung of the sweep. Exponential in
        the dimension, but slowly -- 1.33x per dimension at level 0.9, against 2.68x under the
        median/log(K) bandwidth this code used to use.

        Optimistic for real targets. It is derived on isotropic Gaussians, and anisotropy makes
        matters worse: on three ODE posteriors at dim = 306..608 an ensemble of 400 sat 34-48x
        its Monte Carlo floor under either bandwidth rule. Treat it as a lower bound, use
        stein_R to check, and prefer a fixed `bandwidth` with rescale_stein when dim is large.
        """
        if dim > 12:
            warnings.warn(
                f"min_particles({dim}) extrapolates an exponential fit made over dim = 2..8 "
                f"by a wide margin. Its message -- far more particles than you have -- is "
                f"reliable; its value is not, and should not be quoted as a measurement.",
                stacklevel=2)
        a, b = (8.5, 0.132) if level <= 0.85 else (10.0, 0.287)
        return int(np.ceil(a * np.exp(b * dim)))

    @staticmethod
    def h_star(k, dim):
        """
        Natural scale for a FIXED bandwidth: h* = 2 dim / ln k.

        This is the bandwidth at which the adaptive median heuristic comes to rest, so it is the
        scale a fixed choice should be measured against rather than a recommendation in itself.
        A fixed bandwidth of about 10 h* was the smallest that gave a correctly-dispersed
        attractor on the problems tested; larger values work too and simply take longer to reach
        it, since the trajectory depends on bandwidth x iterations. Requires a small fixed step.
        """
        return 2.0 * dim / np.log(k)

    def stein_ratio(self, particles=None, data=None):
        """
        Stein-identity dispersion ratio of an ensemble. 1 = correctly dispersed, < 1 too narrow.

        Public wrapper over _stein_R that supplies the score itself. Defaults to the ensemble and
        data held on the object, so `m.solve(...); m.stein_ratio()` is the usual call. Named
        distinctly from the `stein_R` attribute that solve() leaves behind, which is this same
        quantity evaluated on the returned ensemble.
        """
        x = self.particles if particles is None else jnp.asarray(particles)
        if x is None:
            raise ValueError("no particles: pass them explicitly or call solve() first")
        d = self.data if data is None else data
        return float(self._stein_R(x, -self.gradient(x, d)))

    def rescale_stein(self, particles=None, data=None):
        """
        Correct a uniform dispersion deficit by rescaling about the ensemble mean by 1/sqrt(R).

        For a Gaussian target and an ensemble of covariance A, R = tr(A Sigma^-1)/dim, so an
        ensemble uniformly a factor c too narrow in variance reads R = c and dividing its
        deviations by sqrt(R) restores the spread. No reference, no tuning, no extra gradient
        evaluations beyond the one that measures R.

        This corrects a SCALE and only a scale. It is worth doing when the collapse is uniform
        across directions, which is what a fixed `bandwidth` in whitened coordinates produces;
        under the adaptive bandwidth the deficit is anisotropic and rescaling will inflate the
        already-adequate directions to fix the deficient ones. It cannot touch an error in the
        ensemble MEAN -- it is applied about that mean.

        Measured with a fixed bandwidth in whitened coordinates on three ODE posteriors: energy
        distance to the reference went from 2.4-6.8x its Monte Carlo floor to 0.57-0.96x.

        returns : (rescaled particles, R measured before rescaling)
        """
        x = self.particles if particles is None else jnp.asarray(particles)
        if x is None:
            raise ValueError("no particles: pass them explicitly or call solve() first")
        d = self.data if data is None else data
        R = self._stein_R(x, -self.gradient(x, d))
        mu = x.mean(axis=0, keepdims=True)
        return mu + (x - mu) / jnp.sqrt(jnp.clip(R, min=1e-12)), float(R)

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
        
    def pairwise_distance(self, particles, h=-1, adaptive=None):
        """
        (squared pairwise distances, bandwidth).

        adaptive : whether to compute the median heuristic. None infers it from h when h is a
            concrete number. Pass it explicitly (a Python bool, static under jit) when h is a
            traced value: otherwise `jnp.where(h <= 0, median, h)` has to evaluate BOTH branches,
            and the median is a sort over k(k-1)/2 entries that is then discarded. That is about
            a quarter of the cost of an SVGD step at k = 400, paid on every iteration of a run
            that uses a fixed bandwidth -- which is the configuration worth using.
        """
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
        if adaptive is None:
            adaptive = not (isinstance(h, (int, float)) and h > 0)
        if not adaptive:
            return L2sq, jnp.asarray(h, particles.dtype)
        upper_tri = np.triu_indices(k, k=1) # upper triangle, excluding diagonal
        median = jnp.median(jnp.clip(L2sq[upper_tri], min=jnp.array(1e-6, dtype=particles.dtype)))
        return L2sq, jnp.where(h <= 0, median, h) # (1,)

    @partial(jax.jit, static_argnames=['self', 'adaptive'])
    def _svgd_update(self, particles, raw_grad, h=-1, adaptive=None):
        '''
        Standard SVGD drift-minus-repulsion update, from the joint RBF kernel.

        raw_grad : (k, dim) -- raw_grad = -self.gradient(particles, data_batch)
        returns  : (k, dim) combined update, fed to a descent optimizer
        '''
        L2sq, h = self.pairwise_distance(particles, h, adaptive)

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
        'grad_clip_enabled', 'monitor_enabled', 'adaptive_bandwidth'])
    def _run_phase(
        self, particles, data, key, *,
        opt_kwargs_values, grad_clip_value, atol, rtol, bandwidth, max_iter, phase, monitor_interval,
        optimizer, opt_kwargs_keys, is_MAP, batch_size, grad_clip_enabled, monitor_enabled,
        adaptive_bandwidth,
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
                grad_particles = self._svgd_update(particles, grad_particles, h=bandwidth,
                                                   adaptive=adaptive_bandwidth)

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
            (see stein_ratio; -> 1 under the target, < 1 underdispersed) every N iterations
            (0 = print status after each phase, < 0 = fully silence). The diagnostic is also
            left on self.stein_R regardless, evaluated on the returned ensemble.

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
        bandwidth           : RBF bandwidth. -1 uses the plain median heuristic, which is
            adaptive and collapses the ensemble variance in high dimensions (see the module
            docstring). A large FIXED value removes that: see h_star for the scale, use about
            10x it, pair it with a small fixed step rather than an adaptive optimizer, and
            finish with rescale_stein.
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
        bandwidth_static  = _listify(bandwidth, n_phases)          # Python values, for the
        bandwidth         = _listify(bandwidth, n_phases, particles.dtype)   # static branch
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
                monitor_enabled=monitor_enabled,
                adaptive_bandwidth=bool(bandwidth_static[i] <= 0),
            )

            if monitor_convergence >= 0:
                max_grad = float(jnp.abs(grad_particles).max())
                print(f"Split {i} finished after {int(n_iter)} iterations | "
                      f"max grad = {max_grad:.5f} | Stein R = {float(stein_R):.4f}")

            # Direct-jump split after every phase but the last
            if i < len(k_schedule):
                particles = self._mitotic_split(particles, key_mitosis, is_MAP_i, k_schedule[i])

        self.particles = particles.copy()
        # Recomputed rather than taken from the loop, which reports the ratio at the positions
        # BEFORE its final update -- one step stale, and confusing next to stein_ratio().
        self.stein_R = self.stein_ratio()
        return self.particles
