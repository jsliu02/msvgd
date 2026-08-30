'''
Alternative implementations of a particle-splitting step for MSVGD, investigated while
deciding what msvgd.py's own MSVGD._mitotic_split should do.

IMPORTANT -- this file is NOT wired into msvgd.py and is no longer a drop-in replacement
mechanism for the production `_mitotic_split`. Following the investigation documented below,
msvgd.py's `_mitotic_split` was rewritten to directly, permanently implement the
covariance-matched-jitter approach (the split_covmatched logic below), with two further
production-specific changes that don't apply to the exploratory variants here:
  1. `solve()` no longer takes `mitosis_splits`/`split_method` -- it takes `k_final` instead
     (None = no split; otherwise one direct jump from the starting particle count to
     k_final), so there is at most one split per solve() call rather than an arbitrary
     sequence of doublings.
  2. New offspring are assigned to a source particle by sampling uniformly at random with
     replacement, rather than each existing particle producing exactly one (or a fixed
     number of) copies -- this supports any k_final greater than the starting count, not
     just exact multiples.
This file remains useful as: (a) the record of *why* covariance-matched jitter was chosen
over the alternatives (see SUMMARY OF FINDINGS below), and (b) reference implementations of
the other strategies considered, if you want to experiment further -- but reusing them now
requires adapting to msvgd.py's current `_mitotic_split(self, particles, key, is_MAP,
k_final)` signature rather than the `make_mitotic_split`/`split_method` mechanism described
below, which no longer exists in msvgd.py.

Each variant below has the signature

    def split_<name>(self, particles, key, is_MAP, budget) -> offspring

and computes only the offspring array (not the concatenation with the existing particles, and
not the jitter-scale budget itself) -- both of those are identical across every variant, so
they're factored into make_mitotic_split() below rather than repeated in each function. This
wrapper (and hence every variant below) still assumes exact doubling, unlike the production
`_mitotic_split`'s arbitrary-k_final design:

    from msvgd.mitotic_split_variants import make_mitotic_split, split_antithetic
    doubling_antithetic_split = make_mitotic_split(split_antithetic)  # for experimentation only

`budget` is the target total expected squared displacement (E[||offspring - parent||^2]),
resolved once per call from the SVGD kernel bandwidth (h/2, matching the kernel's own implicit
Gaussian variance -- see split_isotropic's docstring) or a fixed fallback under is_MAP. Passing
it in rather than letting each variant recompute it from scratch means every variant is
budget-matched by construction: a new variant can only end up with a different total
displacement scale by deliberately ignoring the argument (as split_antithetic does), not by
accident.

`is_MAP` is still passed through explicitly (rather than being fully absorbed into `budget`)
because two variants (split_gradstep, split_density_adaptive) need it for a second reason
beyond budget-sizing: whether to use the SVGD kernel's repulsion term at all, which doesn't
exist in MAP mode.

===========================================================================================
SUMMARY OF FINDINGS (real-data comparison: MAGI FitzHugh-Nagumo inference, 161 discretization
points, ~325-dimensional particle space, mitosis_splits=2 growing k 200->400->800, Prodigy
optimizer with the k-scaling fix applied). Coverage numbers are B=100 independent simulated
datasets unless noted otherwise (an earlier B=10 pass gave the same qualitative picture; B=100
is what's cited below throughout).
===========================================================================================

1. Given a GENEROUS iteration budget per phase (2000 iterations; the no-mitosis condition uses
   a flat k=800 run with iteration count WALL-CLOCK-matched to the 3-phase mitosis run), all 7
   conditions (6 split methods + no-mitosis-at-matched-compute) converge to statistically
   indistinguishable final posteriors:
       isotropic 51.0%, gradstep 49.0%, precond 52.0%, antithetic 51.0%, covmatched 50.0%,
       density_adaptive 52.0%, no_mitosis_matched 51.0%   (joint coverage, all within noise)
   CI widths are similarly within ~3% of each other (0.1616-0.1663). SVGD's own dynamics have
   enough time to "erase" whatever initial configuration a given split method (or skipping
   splitting entirely) produces. At this budget, NEITHER the split method NOR mitosis itself
   matters for final quality.

2. Given a TIGHT iteration budget per phase (500 iterations; no-mitosis condition uses a flat
   k=800 run with iteration count FLOP-matched -- not wall-clock-matched -- to the 3-phase
   mitosis run), real differences emerge:
       covmatched 47.0%, antithetic 46.0%, isotropic 45.0%, precond 43.0%,
       density_adaptive 43.0%, no_mitosis_matched 42.0%, gradstep 39.0%   (joint coverage)
   - split_gradstep is clearly the worst (39.0% joint, and the narrowest/most overconfident CI
     of any condition, 0.1389). Confirmed at B=100 after first showing up at B=10. Not
     recommended.
   - split_covmatched and split_antithetic are the best performers, consistent with their
     faster post-split convergence (see #3) now translating into measurably better final
     coverage once there isn't enough budget left for slower methods to catch up.
   - no_mitosis_matched drops to 42.0% -- near the bottom, behind isotropic/antithetic/
     covmatched. Under a REALISTIC, compute-constrained budget, mitosis (with a good split
     method) is a genuinely better use of a fixed compute budget than running flat SVGD at the
     target k from the start: cheap early iterations while particles are still far from
     converged, expensive iterations only once finer resolution is actually needed.
   Conclusion: finding #1 ("split method / mitosis don't matter") is an artifact of a budget
   generous enough to erase real differences. Under a tight, realistic budget, both the split
   method AND whether you use mitosis at all genuinely matter; the ranking is
   covmatched > antithetic > isotropic > {precond, density_adaptive} > no_mitosis > gradstep.

   FLOP-matching methodology (used for the no-mitosis condition in #2, as distinct from #1's
   wall-clock matching): wall-clock ratios at small k are contaminated by fixed per-phase
   dispatch/compile overhead that isn't part of the actual compute, and fitting
   cost(k) = c1*k + c2*k^2 directly from k=200/400/800 timings gave a nonsensical negative
   quadratic coefficient for exactly this reason. Fitting the same model from LARGER k
   (800/1600/3200/6400, where fixed overhead is a smaller fraction of the total and the O(k)
   gradient-step / O(k^2) SVGD-kernel scaling is cleanly resolved) and extrapolating back down
   gave a clean fit (predictions within ~1-2% of measured at k=1600/3200/6400) and a
   FLOP-matched flat-k=800 iteration count of 820 for the 500-iters/phase budget.

3. Convergence SPEED (residual max-gradient immediately after each split, generous-budget
   single-dataset run) differs clearly even where final quality does not (at that budget):
       split_isotropic:    0.99 -> 0.96   (baseline)
       split_gradstep:      1.58 -> 1.86   (slowest -- worse than baseline)
       split_precond:       0.97 -> 0.77   (modest improvement)
       split_antithetic:    0.27 -> 0.45   (fastest by a wide margin)
       split_covmatched:    0.53 -> 0.31   (fast)
       split_density_adaptive: 0.89 -> 1.06 (no real difference from baseline)
   This convergence-speed ordering is exactly what predicts the tight-budget coverage ranking
   in #2 -- the fast-converging methods are the ones that hold up once the budget gets tight.

4. split_antithetic has a demonstrated CATASTROPHIC failure mode on multimodal, asymmetric-
   weight targets: on a synthetic bimodal 1D target (dominant mode weight 0.9 at x=-5, minor
   mode weight 0.1 at x=+20), reflecting minor-mode particles through the global ensemble mean
   (~-2.5) sent them to x~-25 -- roughly 20 standard deviations from the *nearest* real mode,
   not merely "between" the two modes. 86% of antithetic offspring landed in this zero-density
   region vs. 0% for split_isotropic. This is a structural property (offset = mean - particle,
   unbounded and shape-dependent), not a tuning issue -- see split_antithetic's docstring.

5. IMPORTANT CAVEAT on what "coverage" and "CI width" mean above: a NUTS (gold-standard MCMC)
   baseline run on the same B=100 simulated datasets showed joint coverage of only 30.0% --
   WORSE than every single SVGD configuration above, including every tight-budget one. This
   was cross-checked against an independent full-fidelity NUTS run (1000 warmup + 9000
   samples) on the real (non-simulated) dataset before being trusted, and the two agreed
   closely (theta mean/std within ~2-8% of each other), so it is not a reduced-NUTS-settings
   artifact. NUTS's per-parameter coverage was [99%, 46%, 45%] with the WIDEST CI of anything
   measured (0.2056) -- wide intervals with poor coverage point to BIAS, not insufficient
   spread, as NUTS's failure mode on the b/c parameters specifically. Since NUTS samples the
   true posterior correctly, this means the posterior itself is not well frequentist-
   calibrated for b/c in this particular MAGI+FHN model/noise/sample-size regime -- likely a
   genuine identifiability or prior-interaction issue, not a sampler defect. Practically: the
   split-method rankings in #1/#2 above still hold on their own terms (they're relative
   comparisons among SVGD configurations on identical problems), but a "CI width as a fraction
   of NUTS's CI width" framing (used earlier in this investigation to justify e.g. the
   Fisher-preconditioned kernel and increasing k) should not be read as "as a fraction of the
   width needed for good coverage" -- NUTS's own width doesn't achieve that either, for b/c.

Practical recommendation from this investigation: split_covmatched remains the best general-
purpose candidate and is the current MSVGD/MAGI default -- it is tied-for-best or outright
best under both budgets tested (#1, #2), captures most of antithetic's convergence-speed
advantage while remaining a *local* perturbation anchored to each particle (safe under
multimodality, unlike antithetic, per #4). split_antithetic is worth using only when the
posterior is known/strongly believed to be unimodal and reasonably elliptical, and the
iteration budget is tight enough that its speed advantage matters. split_gradstep (the
original PyTorch/reference MAGI approach) is the one method with a measured downside and no
measured upside relative to the baseline, now confirmed at B=100 across two different
budgets -- avoid it.
'''

import jax
import jax.numpy as jnp
from functools import partial


def make_mitotic_split(split_fn):
    '''
    Wrap a split_xxx(self, particles, key, is_MAP, budget) -> offspring function into a
    complete, jit-compiled _mitotic_split(self, particles, key, is_MAP) -> particles method:
    resolves the shared displacement budget once (from the SVGD kernel bandwidth, or a fixed
    fallback under is_MAP) and concatenates the offspring with the existing particles --
    the two things every variant needs but none of them differ on.
    '''
    @partial(jax.jit, static_argnames=['self', 'is_MAP'])
    def _mitotic_split(self, particles, key, is_MAP):
        if not is_MAP:
            _, h = self.pairwise_distance(particles, -1)
            budget = h / 2
        else:
            budget = 0.01 / 2
        offspring = split_fn(self, particles, key, is_MAP, budget)
        return jnp.concatenate([particles, offspring], axis=0)
    return _mitotic_split


def split_isotropic(self, particles, key, is_MAP, budget):
    '''
    Current production baseline. Each offspring is its parent plus independent, isotropic
    Gaussian jitter, variance budget/dim per coordinate -- giving
    E[||offspring - parent||^2] = budget = h/2, where h is the SVGD kernel's median-heuristic
    bandwidth (h/2 matches the kernel's own implicit Gaussian variance: the kernel is
    exp(-d^2/h), so h = 2*sigma^2 in the usual Gaussian-exponent convention).

    Pros:
      - Simple, well-tested. The former shipped default (split_covmatched is now the default,
        see SUMMARY above), and still a perfectly reasonable choice.
      - Purely LOCAL: each offspring is anchored to its own parent, so the perturbation's
        magnitude and direction never depend on the global shape of the ensemble. Safe
        regardless of whether the target posterior is unimodal, multimodal, symmetric, or
        skewed -- there is no mechanism by which it can throw a particle into a
        disconnected, zero-density region.
      - No extra gradient evaluations or matrix factorizations needed beyond the pairwise
        distance computation already required for the bandwidth heuristic.

    Cons:
      - Ignores curvature/scale heterogeneity across coordinates. Measured Fisher-diagonal
        curvature range in the MAGI test problem was ~1e5 between the stiffest and flattest
        directions; isotropic jitter treats all of them identically.
      - Measurably slower to re-converge after a split than split_antithetic or
        split_covmatched (see SUMMARY above). At a generous iteration budget this doesn't
        translate into worse final coverage/CI width (B=100: 51.0% joint, tied with
        everything but gradstep); at a tight budget it's noticeably behind the two faster
        methods (B=100: 45.0% joint vs. covmatched's 47.0%/antithetic's 46.0%), though still
        clearly better than split_gradstep (39.0%) or skipping mitosis entirely (42.0%).
    '''
    dim = particles.shape[1]
    return particles + jax.random.normal(key, shape=particles.shape, dtype=particles.dtype) * jnp.sqrt(budget / dim)


def split_gradstep(self, particles, key, is_MAP, budget):
    '''
    Reference-style split, matching the original PyTorch/NumPy MAGI implementation: no
    randomness at all. Each offspring is its parent plus one SVGD-combined gradient-ascent
    step on the log-density, magnitude sqrt(budget) -- matching split_isotropic's *expected*
    displacement norm -- so the comparison isolates "random vs. directed displacement"
    rather than differing in overall step size.

    Pros:
      - Deterministic: no fresh random key needed, fully reproducible given the particle
        state alone.
      - Conceptually simple ("nudge the copy toward higher density") and matches the
        pre-existing reference behavior this codebase was ported from.

    Cons:
      - Empirically the WORST of the 6 methods tested, confirmed at B=100 across two
        different iteration budgets. Slowest to re-converge after a split in every trial
        (residual gradient roughly 2x higher than baseline after 2000 iterations), and the
        worst final coverage under a tight, realistic iteration budget: 39.0% joint coverage
        vs. 42-47% for every other condition (including skipping mitosis entirely), with the
        narrowest/most overconfident CI of any condition tested (0.1389). At a generous
        budget the gap narrows (49.0% vs. 50-52% for everything else) but never favors it.
      - Likely mechanism: since every offspring moves in (approximately) the same direction
        as its own parent, parent/offspring pairs end up highly correlated rather than
        diverse. The SVGD repulsion term gets a low-diversity signal to work with right after
        the split, which is exactly when diversity matters most.
      - Not recommended based on this investigation's results.
    '''
    k = particles.shape[0]
    raw_grad = -self.gradient(particles, self.data)
    if not is_MAP:
        kxy, dxkxy = self._svgd_kernel(particles, -1)
        combined = (kxy @ raw_grad - dxkxy) / k
    else:
        combined = raw_grad
    direction = -combined
    unit_dir = direction / (jnp.linalg.norm(direction, axis=1, keepdims=True) + 1e-12)
    return particles + jnp.sqrt(budget) * unit_dir


def split_precond(self, particles, key, is_MAP, budget):
    '''
    Anisotropic jitter using a Fisher-diagonal curvature proxy: per-coordinate variance is
    scaled by 1/fisher_diag, where fisher_diag is the mean squared gradient across the
    current ensemble (a Gauss-Newton-style diagonal curvature estimate), recomputed fresh at
    every split from the particles at that point. Weights are renormalized so the total
    variance budget (summed across coordinates) still matches `budget` -- only the
    *distribution* of that budget across coordinates changes, not its total size.

    This mirrors the diagonal Mahalanobis-preconditioned SVGD kernel investigated earlier in
    this project (which gave a real, reproducible ~4-5 percentage point coverage
    improvement over the isotropic kernel) -- same idea, applied to the splitting step
    instead of the repulsion kernel.

    Pros:
      - Directly targets the measured curvature heterogeneity (fisher_diag range ~1e5 in the
        MAGI test problem) that split_isotropic ignores: jitters more along flat/
        under-constrained directions, less along stiff/well-identified ones.
      - Still a LOCAL perturbation anchored to each particle -- same multimodal-safety
        property as split_isotropic, unlike split_antithetic.
      - Modest convergence-speed improvement over baseline (0.77 vs. 0.96 residual gradient
        after the second split in the single-dataset test).

    Cons:
      - Requires one extra gradient evaluation per split (the fisher_diag estimate), which
        split_isotropic and split_antithetic do not need.
      - Confirmed at B=100 NOT to be among the top performers, despite the theoretical
        motivation and the single-dataset convergence-speed edge over baseline: 52.0% joint
        coverage at the generous budget (statistically tied with everything but gradstep) and
        43.0% at the tight budget -- behind isotropic (45.0%), antithetic (46.0%), and
        covmatched (47.0%), roughly tied with split_density_adaptive. The per-coordinate
        curvature-heterogeneity story is real (see the analogous kernel-preconditioning
        result, a confirmed ~4-5 point coverage improvement), but applying it to the
        *splitting* step specifically doesn't reproduce that benefit -- possibly because the
        fisher_diag estimate from a single, still-early-in-optimization particle ensemble is
        too noisy to usefully redirect the jitter at split time, unlike the kernel case where
        preconditioning is applied continuously throughout optimization.
      - Adds implementation complexity relative to the simpler methods, for a benefit that
        didn't materialize here.
    '''
    dim = particles.shape[1]
    raw_grad = -self.gradient(particles, self.data)
    fisher_diag = jnp.mean(raw_grad**2, axis=0) + 1e-8
    w = 1.0 / jnp.sqrt(fisher_diag)
    w_normalized = w / jnp.sqrt(jnp.mean(w**2))
    per_dim_std = jnp.sqrt(budget / dim) * w_normalized
    return particles + jax.random.normal(key, shape=particles.shape, dtype=particles.dtype) * per_dim_std


def split_antithetic(self, particles, key, is_MAP, budget):
    '''
    Deterministic mirrored duplication: each offspring is its parent reflected through the
    current ensemble mean, offspring = 2*mean - parent. Ignores both `is_MAP` and `budget`:
    no jitter scale to tune, no randomness needed.

    Pros:
      - Fastest observed convergence after a split by a wide margin in this investigation
        (residual gradient 0.27 -> 0.45 across two splits, vs. 0.99 -> 0.96 for baseline).
        That speed advantage is confirmed to matter at B=100 under a tight iteration budget:
        46.0% joint coverage, second only to split_covmatched (47.0%) and clearly ahead of
        split_isotropic (45.0%) and everything else.
      - Zero hyperparameters: no bandwidth-derived scale, no covariance estimate.
      - Guarantees the offspring is maximally "spread" relative to the parent for a
        genuinely unimodal, roughly symmetric/elliptical posterior -- there is no chance of
        an unlucky small random draw producing a near-duplicate pair, which random-jitter
        methods can occasionally do.

    Cons -- DEMONSTRATED, NOT JUST THEORETICAL:
      - Catastrophic failure mode on multimodal, asymmetric-weight targets. The offset for a
        given particle is (mean - particle): proportional to that particle's distance from
        the *global* ensemble mean, with NO upper bound and no dependence on where actual
        density is. Tested directly on a synthetic 1D bimodal target (dominant mode weight
        0.9 at x=-5, minor mode weight 0.1 at x=+20; ensemble mean ~-2.7): reflecting the
        minor-mode particles sent them to x~-25 to -27 -- about 20 standard deviations from
        the *nearest* real mode (mode A), not merely into the gap between the two modes.
        86% of antithetic offspring landed in this zero-density "no-man's-land" region,
        vs. 0% for split_isotropic on the identical ensemble.
      - In a live optimization (unlike this static snapshot), particles landing that far
        into a zero-density region would produce a destructive, oversized gradient on the
        very next step -- the same class of failure this codebase's earlier
        mitosis-jitter-dimensionality bug produced, just via a different mechanism.
      - Risk compounds under repeated application: if applied at every mitosis split, an
        already-symmetric-ish ensemble can develop "mirror duplicate" structure (pairs that
        are near-exact reflections of each other) rather than genuinely novel spatial
        coverage, reducing effective diversity over multiple splits.
      - Also degrades for heavily skewed unimodal distributions, where the mean is not a
        representative central point of the bulk of the density (it gets pulled toward the
        tail), and for very weakly-identified/ridge-like directions, where doubling the
        distance-from-mean can overshoot well past the region the data actually supports.

    Use only when the posterior is known or strongly believed to be unimodal and reasonably
    elliptical. Not recommended as a general-purpose default for arbitrary user-supplied ODE
    models, where multimodality cannot be ruled out in advance.
    '''
    mean = jnp.mean(particles, axis=0, keepdims=True)
    return 2 * mean - particles


def split_covmatched(self, particles, key, is_MAP, budget):
    '''
    Jitter drawn from a multivariate Gaussian fit to the current ensemble's own empirical
    covariance (a smoothed-bootstrap-style perturbation), rescaled so the total variance
    budget (trace of the scaled covariance) matches `budget`, the same budget used by
    split_isotropic and split_precond.

    Pros:
      - This is the MSVGD/MAGI production default (see msvgd.py's `_mitotic_split` and
        `solve()`'s `split_method` argument), based on the results below.
      - Second-fastest convergence in the single-dataset check, close to split_antithetic
        (0.53 -> 0.31 residual gradient across two splits) and clearly better than baseline.
        At B=100, this translates into the BEST joint coverage of every condition tested
        under a tight, realistic iteration budget (47.0%, vs. 46.0% for split_antithetic and
        45.0% for split_isotropic), and is statistically tied for best at the generous budget
        (50.0%, indistinguishable from the 49-52% range of every other condition there).
      - Adapts automatically to the current shape and cross-coordinate correlation structure
        of the ensemble -- no manual per-coordinate tuning needed, unlike split_precond's
        diagonal-only approximation (which, unlike this method, did NOT show a tight-budget
        coverage benefit despite a similar theoretical motivation -- see its docstring).
      - Remains a LOCAL perturbation anchored to each particle (offspring = particle + zero-
        mean noise), not a reflection through a global point -- structurally much safer than
        split_antithetic under multimodality. A particle at a minor mode still just gets a
        smear around its own position, never a deterministic launch toward a specific
        far-off point.

    Cons:
      - Requires computing a (dim x dim) empirical covariance and its Cholesky factor at
        every split: O(k*dim^2) for the covariance, O(dim^3) for the factorization. Trivial
        at MAGI's ~325-dimensional scale on GPU, but would need attention at much higher
        dimensionality.
      - Not perfectly immune to multimodality: the empirical covariance is still a *global*
        ensemble statistic, and if the ensemble spans well-separated modes, that covariance
        can be inflated by the between-mode spread (making jitter too large even for
        within-mode dynamics). This is a real but much milder version of split_antithetic's
        problem -- it stays a zero-mean local perturbation around each particle's own
        position rather than a deterministic, unbounded reflection, so it cannot by itself
        throw a particle into a disconnected zero-density region the way antithetic does.
      - Based on the evidence gathered here, this is the strongest general-purpose
        alternative to the isotropic baseline: most of the speed benefit, none of
        antithetic's demonstrated catastrophic failure mode.
    '''
    k, dim = particles.shape
    mean = jnp.mean(particles, axis=0, keepdims=True)
    centered = particles - mean
    cov = (centered.T @ centered) / k
    cov = cov + 1e-6 * jnp.eye(dim, dtype=particles.dtype)
    L = jnp.linalg.cholesky(cov)
    z = jax.random.normal(key, shape=particles.shape, dtype=particles.dtype)
    raw_jitter = z @ L.T
    scale = jnp.sqrt(budget / jnp.trace(cov))
    return particles + scale * raw_jitter


def split_density_adaptive(self, particles, key, is_MAP, budget):
    '''
    Density-adaptive resampling, borrowed from adaptive-resampling ideas in particle
    filters/SMC: rather than duplicating every particle exactly once, k parents are drawn
    WITH REPLACEMENT, weighted by 1/(local kernel density), so particles in sparse
    (under-represented) regions of the ensemble are preferentially selected as split parents
    over particles in dense regions. A small isotropic jitter (same budget as
    split_isotropic) is then added to break exact-duplicate ties when the same parent is
    drawn more than once.

    Pros:
      - Directly targets representation quality rather than treating every existing particle
        as equally deserving of a copy -- conceptually the most targeted attempt at fixing
        genuine under-sampling.
      - Uses only LOCAL density estimates (kernel similarity to neighbors) rather than a
        global mean or covariance, so it does not inherit split_antithetic's specific
        "reflect through a potentially meaningless point" failure mode.

    Cons:
      - Empirically, no advantage over split_isotropic was found in this investigation, and
        some indication of a mild disadvantage: convergence speed after a split was
        essentially identical to baseline in the single-dataset check (0.89 -> 1.06 vs.
        baseline's 0.99 -> 0.96), coverage/CI width were indistinguishable at the generous
        B=100 budget (52.0% vs. 51.0% joint), but at the tight B=100 budget it came in behind
        split_isotropic (43.0% vs. 45.0% joint, tied with split_precond). The added
        complexity did not pay off on this problem.
      - Resampling with replacement means the doubling is no longer a strict 1:1 parent ->
        offspring mapping: some particles can be selected 0, 1, or several times, which can
        reduce effective sample size in already-sparse regions if the density weighting is
        noisy (density estimated from only k particles can itself be a poor estimate,
        especially early in optimization before the ensemble has organized around the
        target).
      - Introduces a discrete sampling step (jax.random.choice) and an extra kernel
        evaluation per split, more implementation complexity than split_isotropic for no
        measured benefit on the problem tested here. Might still be worth revisiting on a
        problem with genuinely uneven/patchy posterior coverage where density really is
        badly non-uniform across the existing ensemble, which was not strongly the case in
        the MAGI FHN test problem.
    '''
    k, dim = particles.shape
    if not is_MAP:
        kxy, _ = self._svgd_kernel(particles, -1)
        local_density = jnp.sum(kxy, axis=1)
        weights = 1.0 / (local_density + 1e-6)
        weights = weights / jnp.sum(weights)
    else:
        weights = jnp.ones(k, dtype=particles.dtype) / k
    key1, key2 = jax.random.split(key)
    idx = jax.random.choice(key1, k, shape=(k,), p=weights, replace=True)
    selected = particles[idx]
    jitter = jax.random.normal(key2, shape=particles.shape, dtype=particles.dtype) * jnp.sqrt(budget / dim)
    return selected + jitter
