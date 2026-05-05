import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from functools import partial
from collections.abc import Iterable
import inspect

def _listify(val, length, dtype=None):
    '''
    Helper function to prepare a numerical/iterable argument for mitosis splits.
    Not user-facing.
    '''
    if isinstance(val, Iterable) and type(val) is not dict:
        if len(val) == length:
            listed = val
        else:
            raise ValueError(
                f"Incorrect gradient descent hyperparameter argument length, "
                f"got {len(val)}, expecting {length}."
            )
    else:
        listed = [val] * length

    if dtype is not None:
        return jnp.array(listed, dtype=dtype)
    else:
        return listed

class MSVGD():
    def __init__(self, logdensity, data=None):
        '''
        Define log-density of the target distribution, may be up to additive constant.

        If solve will used batched gradient descent, should have signature logdensity(params, data_batch).
        '''
        self.data = data

        if len(inspect.signature(logdensity).parameters) == 2:
            # logdensity must accept (x, data_batch)
            def _single_grad(x, data_batch):
                return jax.grad(lambda x: logdensity(x, data_batch).sum())(x)
            self.gradient = jax.jit(jax.vmap(_single_grad, in_axes=(0, None)))
            self._batched = True
        elif len(inspect.signature(logdensity).parameters) == 1:
            _single_grad = jax.grad(lambda x: logdensity(x).sum())
            self.gradient = jax.jit(jax.vmap(_single_grad, in_axes=0))
            self._batched = False
        else:
            raise ValueError("The logdensity has an invalid number of arguments (1 or 2 if data batching).")
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
        log_k = jnp.log(jnp.array(k, dtype=particles.dtype))
        upper_tri = jnp.triu_indices(k, k=1) # keep upper triangle, excluding diagonal
        h = jnp.where(h <= 0, jnp.median(jnp.clip(L2sq[upper_tri], jnp.array(0.0, dtype=particles.dtype))) / log_k, h) # (1,)

        Kxy = jnp.exp(-L2sq / h) # (k, k)
        dxkxy = (Kxy.sum(axis=1, keepdims=True) * particles - Kxy @ particles) * (jnp.array(2.0, dtype=particles.dtype) / h) # (k, d)

        return Kxy, dxkxy

    def _mitotic_split(self, particles, key):
        '''
        Double the particle count by concatenating the current particles with a jittered copy.
        In JAX particles are immutable arrays; we return the new array.
        Not JIT-compiled because it is called with a different `particles` shape each time.
        '''
        # empirical std of each dimension across current particles (k, d) -> (d,)
        stds = jnp.std(particles, axis=0).clip(jnp.array(1e-6, dtype=particles.dtype))   # (d,) — one scale per dimension
        jitter = jax.random.normal(key, shape=particles.shape, dtype=particles.dtype) * stds

        return jnp.concatenate([particles, particles + jitter], axis=0)

    def solve(
        self,
        x0,
        mitosis_splits=0,
        random_seed=8,
        optimizer=optax.adam,
        optimizer_kwargs={"learning_rate": 0.1},
        batch_size=None,
        is_MAP=False,
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
        random_seed         : int used to set jax.random key for sampling mitosis jitters

        Note: The following arguments may each be passed as a single value to be used globally
            or as a list of length `mitosis_splits+1`, containing (different) values for each mitosis phase.
        optimizer           : an optax optimizer constructor, or list thereof, configured for descent
        optimizer_kwargs    : dict of kwargs passed to the optimizer, or list thereof
            Warning : It is necessary in some case for optimizer kwargs to have the same dtype as x0,
                e.g. {"learning_rate" : jnp.array(0.1, dtype=x0.dtype)}
        batch_size          : int or list of ints (one per phase) for stochastic optimization, None for full dataset
        is_MAP              : bool whether to mode-find using on the gradient of only the logdensity
        max_iter            : int or list of ints (one per phase)
        atol, rtol          : convergence tolerances,  all(grad <= atol + rtol * particles)
        bandwidth           : RBF bandwidths (-1 = median heuristic)

        monitor_convergence : int — print max grad every N iterations
            (0 = print status after each mitosis split, < 0 = fully silence)
        '''
        key = jr.PRNGKey(random_seed)
        n_phases = mitosis_splits + 1

        optimizer        = _listify(optimizer, n_phases)
        optimizer_kwargs = _listify(optimizer_kwargs, n_phases)
        batch_size       = _listify(batch_size, n_phases)  # None means full batch
        max_iter         = _listify(max_iter, n_phases)
        atol             = _listify(atol, n_phases, x0.dtype)
        rtol             = _listify(rtol, n_phases, x0.dtype)
        bandwidth        = _listify(bandwidth, n_phases, x0.dtype)
        if self._batched:
            N = self.data.shape[0]

        # Ensure that particles are a JAX array
        # Tsyping will carry over if x0 was originally passed as a JAX array
        particles = jnp.array(x0)

        for i in range(n_phases):
            k = particles.shape[0]
            if k == 1:
                is_MAP = True  # no SVGD kernel if doing MAP estimation
            k = jnp.array(k, dtype=particles.dtype)

            batch_size_i = batch_size[i]
            if self._batched and batch_size_i is not None:
                n_batches = N // batch_size_i
                key, subkey = jr.split(key)
                perm = jr.permutation(subkey, N)
                data_shuffled = self.data[perm]
    
            bw_i = bandwidth[i]
            atol_i = atol[i]
            rtol_i = rtol[i]
            mc = monitor_convergence

            opt = optimizer[i](**optimizer_kwargs[i])
            opt_state = opt.init(particles)


            # ------------------------------------------------------------------
            # Inner step: one gradient + optimizer update.
            # Captured variables (gradient, opt, is_MAP, k, bw_i) are all static
            # From JAX's perspective — they don't change during the while_loop.
            # ------------------------------------------------------------------
            def body_fn(carry):
                particles, opt_state, _, iteration, key = carry
                key, subkey = jr.split(key)
                # --- gradient computation ---
                if self._batched and batch_size_i is not None:
                    batch_start = (iteration % n_batches) * batch_size_i
                    data_batch = jax.lax.dynamic_slice_in_dim(
                        data_shuffled, batch_start, batch_size_i, axis=0)

                    grad_particles = -self.gradient(particles, data_batch)
                else:
                    grad_particles = -self.gradient(particles)

                # Compute SVGD gradient direction
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
                return (particles, opt_state, grad_particles, iteration + 1, key)

            def cond_fn(carry):
                particles, _, grad_particles, iteration, _ = carry
                not_converged = ~jnp.all(
                    jnp.abs(grad_particles) <= atol_i + rtol_i * jnp.abs(particles)
                )
                under_max_iter = iteration < max_iter[i]
                return not_converged & under_max_iter

            # Seed grad with inf so the convergence check always runs at least one step
            key_sgd, key_mitosis = jr.split(jr.fold_in(key, i))

            init_grad = jnp.full_like(particles, jnp.inf)
            init_carry = (particles, opt_state, init_grad, jnp.zeros((), jnp.int32), key_sgd)

            particles, _, grad_particles, n_iter, _ = jax.lax.while_loop(
                cond_fn, body_fn, init_carry
            )

            if mc >= 0:
                max_grad = float(jnp.abs(grad_particles).max())
                print(f"Split {i} finished after {int(n_iter)} iterations | max grad = {max_grad:.5f}")

            # Mitotic split (except after the last phase)
            if i < mitosis_splits:
                particles = self._mitotic_split(particles, key_mitosis)

        self.particles = particles.copy()
        return particles