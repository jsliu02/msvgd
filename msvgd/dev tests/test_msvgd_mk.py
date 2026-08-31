'''
Smoke tests + a documented failure mode for msvgd_mk.MKSVGD, run directly:

    python test_msvgd_mk.py

1. n_kernels=1 equivalence: MKSVGD._mk_svgd_update must exactly reduce to the standard
   _svgd_kernel-based combined update when there's only one kernel in the ladder (weight
   trivially 1.0).
2. demo_wide_ladder_collapse: NOT a pass/fail test -- documents that the paper's own Eq. 21
   weight formula, combined with a wide bandwidth ladder (10 kernels spanning 9 orders of
   magnitude around the median-heuristic bandwidth, the first thing tried), catastrophically
   collapses on a toy correlated 2D Gaussian: the KSD magnitude used for weighting diverges
   as h -> 0, so weight concentrates on the smallest, most degenerate (near-zero-repulsion)
   kernel. The safe defaults (n_kernels=5, ratio=2.0) avoid this.
3. test_toy_gaussian_safe_range: the safe defaults recover a covariance close to standard
   SVGD's on the same toy problem (not catastrophically collapsed).
'''

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from msvgd_mk import MKSVGD
from msvgd import MSVGD


def test_n_kernels_1_equivalence():
    key = jr.PRNGKey(0)
    k, dim = 50, 3
    particles = jr.normal(key, (k, dim), dtype=jnp.float64) * 2.0
    raw_grad = jr.normal(jr.fold_in(key, 1), (k, dim), dtype=jnp.float64)

    model_mk = MKSVGD(lambda x: -0.5 * jnp.sum(x ** 2))
    model_std = MSVGD(lambda x: -0.5 * jnp.sum(x ** 2))

    combined_mk = model_mk._mk_svgd_update(particles, raw_grad, n_kernels=1, ratio=10.0)
    kxy, dxkxy = model_std._svgd_kernel(particles, h=-1)
    combined_std = (kxy @ raw_grad - dxkxy) / k

    max_abs_diff = jnp.max(jnp.abs(combined_mk - combined_std))
    assert max_abs_diff < 1e-8, f"n_kernels=1 mismatch: max abs diff = {max_abs_diff}"
    print(f"[PASS] n_kernels=1 equivalence: max abs diff = {max_abs_diff:.2e}")


def _correlated_gaussian_setup():
    target_mean = jnp.array([1.0, -2.0])
    target_cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
    target_prec = jnp.linalg.inv(target_cov)

    def logdensity(x):
        d = x - target_mean
        return -0.5 * d @ target_prec @ d

    key = jr.PRNGKey(1)
    x0 = jr.normal(key, shape=(200, 2), dtype=jnp.float64) * 0.1
    return logdensity, x0, target_cov


def demo_wide_ladder_collapse():
    logdensity, x0, target_cov = _correlated_gaussian_setup()
    model = MKSVGD(logdensity)
    particles = x0
    opt = optax.adam(0.1)
    opt_state = opt.init(particles)
    for _ in range(3000):
        raw_grad = -model.gradient(particles, None)
        combined = model._mk_svgd_update(particles, raw_grad, n_kernels=10, ratio=10.0)
        updates, opt_state = opt.update(combined, opt_state, particles)
        particles = optax.apply_updates(particles, updates)

    print(f"[INFO] wide ladder (n_kernels=10, ratio=10) empirical cov =\n{jnp.cov(particles.T)}")
    print(f"[INFO] target cov =\n{target_cov}")
    print("[INFO] expected: badly collapsed (variances far below target) -- documented failure mode, not a bug fix target")


def test_toy_gaussian_safe_range():
    logdensity, x0, target_cov = _correlated_gaussian_setup()
    particles = MKSVGD(logdensity).solve(x0, random_seed=0, max_iter=10_000, atol=1e-4, monitor_convergence=-1)
    emp_cov = jnp.cov(particles.T)
    print(f"[INFO] safe-range (n_kernels=5, ratio=2.0) empirical cov =\n{emp_cov}")
    print(f"[INFO] target cov =\n{target_cov}")
    max_var_err = jnp.max(jnp.abs(jnp.diag(emp_cov) - jnp.diag(target_cov)))
    assert max_var_err < 0.5, f"variance error too large (looks collapsed?): {max_var_err}"
    print(f"[PASS] safe-range toy Gaussian: max variance error = {max_var_err:.3f}")


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    test_n_kernels_1_equivalence()
    demo_wide_ladder_collapse()
    test_toy_gaussian_safe_range()
    print("Done.")
