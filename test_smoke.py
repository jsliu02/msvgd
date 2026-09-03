"""
Smoke tests for msvgd. Run directly:  python test_smoke.py

Checks the public surface on targets whose answers are known analytically, so every assertion is
against a number rather than against a previous run. Exits nonzero on any failure.
"""
import os, sys, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msvgd.msvgd import MSVGD

N_PASS, N_FAIL, FAILS = 0, 0, []


def check(name, cond, detail=""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        N_FAIL += 1
        FAILS.append(name)
        print(f"  FAIL  {name}   {detail}")


def raises(name, fn, exc=Exception):
    try:
        fn()
    except exc as e:
        check(name, True, f"{type(e).__name__}")
    except Exception as e:
        check(name, False, f"wrong exception {type(e).__name__}")
    else:
        check(name, False, "no exception raised")


STD = lambda x: -0.5 * jnp.sum(x ** 2)                       # N(0, I), score = -x

# =============================================================== bandwidth
print("\n### pairwise_distance: the plain median, not median/log(k)")
rng = np.random.default_rng(0)
X = jnp.asarray(rng.standard_normal((200, 20)))
m = MSVGD(STD)
L2, h = m.pairwise_distance(X)
med = float(jnp.median(L2[np.triu_indices(200, 1)]))
check("bandwidth is the plain median", abs(float(h) - med) < 1e-9,
      f"h={float(h):.4f} median={med:.4f} median/lnK={med/np.log(200):.4f}")
_, h_fixed = m.pairwise_distance(X, h=7.5)
check("a positive bandwidth overrides the heuristic", abs(float(h_fixed) - 7.5) < 1e-12)
check("L2sq is symmetric with a zero diagonal",
      np.allclose(L2, L2.T, atol=1e-10) and float(jnp.abs(jnp.diag(L2)).max()) < 1e-10)

# =============================================================== sizing helpers
print("\n### min_particles / h_star")
measured = {2: 16, 3: 24, 4: 32, 5: 48, 6: 64, 7: 64, 8: 96}   # exp06b, level 0.9
worst = 0.0
for d, k in measured.items():
    pred = MSVGD.min_particles(d)
    worst = max(worst, abs(np.log(pred / k)))
    check(f"min_particles({d}) is within a sweep rung of {k}", 0.5 * k <= pred <= 2 * k,
          f"predicted {pred}")
check("min_particles agrees with the fit it documents", worst < np.log(2), f"max log-ratio {worst:.3f}")
check("min_particles is monotone in the dimension",
      all(MSVGD.min_particles(d) <= MSVGD.min_particles(d + 1) for d in range(2, 12)))
check("a lower level needs fewer particles", MSVGD.min_particles(8, 0.8) < MSVGD.min_particles(8, 0.9),
      f"{MSVGD.min_particles(8, 0.8)} < {MSVGD.min_particles(8, 0.9)}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    MSVGD.min_particles(325)
    check("min_particles warns when extrapolating far past its fit", len(w) == 1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    MSVGD.min_particles(5)
    check("min_particles is silent inside its fitted range", len(w) == 0)
check("h_star = 2 dim / ln k", abs(MSVGD.h_star(400, 325) - 2 * 325 / np.log(400)) < 1e-9,
      f"{MSVGD.h_star(400, 325):.2f}")
check("h_star grows with dimension", MSVGD.h_star(400, 50) < MSVGD.h_star(400, 500))

# =============================================================== the Stein ratio
print("\n### stein_R: for N(0, I) it must equal the mean variance ratio")
for c in (0.25, 0.5, 1.0, 2.0):
    Xc = jnp.asarray(rng.standard_normal((512, 15)) * np.sqrt(c))
    mm = MSVGD(STD); mm.particles = Xc
    R = mm.stein_ratio()
    v = float(jnp.mean(jnp.var(Xc, axis=0)))
    check(f"stein_R reads the variance ratio at c={c}", abs(R - v) < 1e-9,
          f"R={R:.4f} var={v:.4f}")
mm = MSVGD(STD)
raises("stein_ratio without particles errors", mm.stein_ratio, ValueError)
raises("rescale_stein without particles errors", mm.rescale_stein, ValueError)

print("\n### rescale_stein")
Xn = jnp.asarray(rng.standard_normal((512, 15)) * 0.5 + 3.0)   # 4x too narrow, offset mean
mm = MSVGD(STD); mm.particles = Xn
fixed, R0 = mm.rescale_stein()
check("rescale_stein reports the pre-correction R", abs(R0 - float(jnp.mean(jnp.var(Xn, 0)))) < 1e-9,
      f"{R0:.4f}")
check("rescale_stein brings the variance to 1",
      abs(float(jnp.mean(jnp.var(fixed, 0))) - 1.0) < 1e-9,
      f'{float(jnp.mean(jnp.var(fixed, 0))):.6f}')
check("rescale_stein preserves the ensemble mean",
      float(jnp.abs(fixed.mean(0) - Xn.mean(0)).max()) < 1e-12)
mok = MSVGD(STD)
mok.particles = jnp.asarray(rng.standard_normal((4096, 5)))
ok_fixed, ok_R = mok.rescale_stein()
check("rescale_stein barely moves a correctly-dispersed ensemble", abs(ok_R - 1.0) < 0.05,
      f"R={ok_R:.4f}")
check("  and its correction is correspondingly small",
      float(jnp.abs(ok_fixed - mok.particles).max()) < 0.05 * float(jnp.abs(mok.particles).max()))

# =============================================================== solve()
print("\n### solve: paths and invariants")
X0 = jnp.asarray(rng.standard_normal((64, 8)))
base = dict(k_schedule=[], max_iter=300, optimizer=optax.adam,
            optimizer_kwargs={"learning_rate": 0.05}, monitor_convergence=-1)
mm = MSVGD(STD)
out = mm.solve(x0=X0, **base)
check("solve returns the ensemble shape it was given", out.shape == X0.shape)
check("solve output is finite", bool(jnp.all(jnp.isfinite(out))))
check("solve leaves particles on the object", mm.particles is not None)
check("solve leaves stein_R on the object", isinstance(mm.stein_R, float))
check("the stein_R attribute does not shadow the stein_ratio method",
      abs(mm.stein_R - mm.stein_ratio()) < 1e-9, f"{mm.stein_R:.6f}")
check("solve is deterministic", np.allclose(MSVGD(STD).solve(x0=X0, **base), out))

mm2 = MSVGD(STD)
out2 = mm2.solve(x0=X0, k_schedule=[128, 256], max_iter=200, optimizer=optax.adam,
                 optimizer_kwargs={"learning_rate": 0.05}, monitor_convergence=-1)
check("k_schedule grows the ensemble", out2.shape == (256, 8), f"{out2.shape}")
raises("a non-increasing k_schedule is rejected",
       lambda: MSVGD(STD).solve(x0=X0, k_schedule=[32], max_iter=10, optimizer=optax.adam,
                                optimizer_kwargs={"learning_rate": 0.05},
                                monitor_convergence=-1), ValueError)
raises("a mismatched per-phase hyperparameter list is rejected",
       lambda: MSVGD(STD).solve(x0=X0, k_schedule=[128], max_iter=[10, 10, 10],
                                optimizer=optax.adam,
                                optimizer_kwargs={"learning_rate": 0.05},
                                monitor_convergence=-1), ValueError)

mp = MSVGD(STD).solve(x0=X0, is_MAP=True, max_iter=2000, k_schedule=[], optimizer=optax.adam,
                      optimizer_kwargs={"learning_rate": 0.05}, monitor_convergence=-1)
check("is_MAP collapses every particle onto the mode",
      float(jnp.abs(mp).max()) < 1e-2, f"max |x| = {float(jnp.abs(mp).max()):.2e}")

fx = MSVGD(STD).solve(x0=X0, bandwidth=10 * MSVGD.h_star(64, 8), max_iter=2000,
                      k_schedule=[], optimizer=optax.sgd,
                      optimizer_kwargs={"learning_rate": 0.01}, monitor_convergence=-1)
check("a fixed bandwidth with a small fixed step runs and stays finite",
      bool(jnp.all(jnp.isfinite(fx))))
clip = MSVGD(STD).solve(x0=X0, grad_clip=1.0, **base)
check("grad_clip runs", bool(jnp.all(jnp.isfinite(clip))))

for dt in (jnp.float32, jnp.float64):
    o = MSVGD(STD).solve(x0=jnp.asarray(X0, dt), **base)
    check(f"solve preserves {dt.__name__}", o.dtype == dt)

# =============================================================== the documented failure
print("\n### the collapse is real and the tools see it")
d_hi = 200
Xh = jnp.asarray(rng.standard_normal((64, d_hi)))
mh = MSVGD(STD)
oh = mh.solve(x0=Xh, k_schedule=[], max_iter=1500, optimizer=optax.adam,
              optimizer_kwargs={"learning_rate": 0.05}, monitor_convergence=-1)
v = float(jnp.mean(jnp.var(oh, axis=0)))
check("started at exact draws in d=200, K=64, the variance collapses", v < 0.5, f"var {v:.4f}")
check("stein_R detects it", mh.stein_R < 0.6, f"R {mh.stein_R:.4f}")
check("min_particles said so in advance", MSVGD.min_particles(d_hi, 0.8) > 64,
      f"needs ~{MSVGD.min_particles(d_hi, 0.8):.3g}")
_, Rpre = mh.rescale_stein()
check("rescale_stein agrees with the stored ratio", abs(Rpre - mh.stein_R) < 1e-9,
      f"{Rpre:.6f} vs {mh.stein_R:.6f}")

# =============================================================== batching
print("\n### data batching")
data = jnp.asarray(rng.standard_normal((256, 3)))
logp_b = lambda x, batch: -0.5 * jnp.sum(x ** 2) - 1e-6 * jnp.sum(batch ** 2)
mb = MSVGD(logp_b, data=data)
ob = mb.solve(x0=jnp.asarray(rng.standard_normal((32, 4))), batch_size=64, max_iter=100,
              k_schedule=[], optimizer=optax.adam, optimizer_kwargs={"learning_rate": 0.05},
              monitor_convergence=-1)
check("batched solve runs", ob.shape == (32, 4) and bool(jnp.all(jnp.isfinite(ob))))
raises("a logdensity with the wrong arity is rejected",
       lambda: MSVGD(lambda a, b, c: a), ValueError)

print(f"\n{'=' * 70}\n{N_PASS} passed, {N_FAIL} failed"
      + (f"\nfailures: {FAILS}" if FAILS else ""))
sys.exit(1 if N_FAIL else 0)
