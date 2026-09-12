"""
Environment-level ablation knobs shared by pg_adapt.py / pepg_adapt.py
(and sac_adapt.py through pg_adapt). Two axes, both multiplicative scales
whose value 1.0 reproduces the main campaign exactly:

  wealth-gap scale  s_gap   -- how far apart the two groups START
  performative scale s_perf -- how strongly the population REACTS to
                               the policy's decisions (see the env classes'
                               `performative_scale` for the mechanism)

Only the wealth-gap transform lives here (it acts on the loaded data
before the environment is built); the performative scale is a constructor
argument of IncomeEnvironment / TestingIncomeEnvironment.
"""

import numpy as np


def scale_wealth_gap(X_male, X_female, N_male, N_female, scale):
    """Rescale each group's initial wealth so the initial mean wealth gap
    mu_M - mu_F becomes `scale` times its empirical value, while

      * the population-mean wealth (N-weighted over the N_male + N_female
        individuals the environment actually uses) is UNCHANGED -- so the
        ablation moves inequality, not the overall wealth level, which
        matters because the arrival rate depends on relative standing
        mu_g / mu_bar (environment.MATTHEW_C);
      * each group's within-group distribution keeps its shape (a single
        multiplicative factor per group, so the coefficient of variation
        and the rank order of individuals are preserved);
      * creditworthiness, default probabilities and loan amounts are
        untouched (they come from the fitted TransitionParameterLearner,
        not from X) -- same people, different starting wealth.

    scale = 1 -> identity; 0 -> both groups start at the population mean
    (no initial gap); 2 -> twice the empirical gap. Means are taken over
    the first N_g entries because that is the slice the environment keeps.
    Returns (X_male', X_female') as new float64 arrays (inputs untouched).
    """
    scale = float(scale)
    X_male = np.asarray(X_male, dtype=np.float64).copy()
    X_female = np.asarray(X_female, dtype=np.float64).copy()
    if scale == 1.0:
        return X_male, X_female
    m_M = float(X_male[:N_male].mean())
    m_F = float(X_female[:N_female].mean())
    m_bar = (N_male * m_M + N_female * m_F) / (N_male + N_female)
    t_M = m_bar + scale * (m_M - m_bar)
    t_F = m_bar + scale * (m_F - m_bar)
    if t_M <= 0.0 or t_F <= 0.0:
        raise ValueError(
            f"wealth-gap scale {scale} would push a group's mean wealth to "
            f"<= 0 (targets mu_M={t_M:.2f}, mu_F={t_F:.2f}); empirical "
            f"mu_M={m_M:.2f}, mu_F={m_F:.2f}, mu_bar={m_bar:.2f}")
    X_male *= t_M / m_M
    X_female *= t_F / m_F
    return X_male, X_female
