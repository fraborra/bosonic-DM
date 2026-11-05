// Copyright (C) 2026 Francesco Borra
//
functions {
  // Exact integral of the normalized Primakoff-Rosen density on [0, E].
  real primakoff_rosen_cdf(real E, real Q) {
    real q = Q / 510.99895;
    real z = E / Q;
    vector[5] coefficients = [30, 60, 40, 10, 1]';
    real numerator = 0;
    real denominator = 0;
    if (E <= 0) return 0;
    if (E >= Q) return 1;
    for (k in 0:4) {
      real weight = coefficients[k + 1] * pow(q, k)
                    * exp(lbeta(k + 2.0, 6.0));
      numerator += weight * beta_cdf(z | k + 2.0, 6.0);
      denominator += weight;
    }
    return numerator / denominator;
  }

  real modified_beta_g_cdf(real E, real mu, real sigma, real tau,
                           real base_alpha, real base_beta,
                           real loc, real scale) {
    real u;
    real r;
    if (E <= loc) return 0;
    if (E >= loc + scale) return 1;
    u = beta_cdf((E - loc) / scale | base_alpha, base_beta);
    // Explicit endpoints avoid undefined derivatives at saturated CDF values.
    if (u <= 0) return 0;
    if (u >= 1) return 1;
    r = inv_logit(log(tau) + logit(u));
    return beta_cdf(r | mu, sigma);
  }
}
data {
  int<lower=1> n_bins;
  vector<lower=0>[n_bins + 1] edges; // keV; strictly increasing, <= Q
  array[n_bins] int<lower=0> counts;
  // 1 = bin enters the likelihood; 0 = excluded (e.g. gamma-peak window).
  array[n_bins] int<lower=0, upper=1> use_bin;
  real<lower=0> Q;
  real loc;                       // keV
  real<lower=0> scale;             // keV
  // Fixable parameters: 1 = sampled; 0 = fixed to the value below, removed
  // from the sampled space, and without a prior (its prior entries are
  // ignored). Fixing c0 = cQ = 0 gives the model without the linear term.
  int<lower=0, upper=1> fit_c0;
  int<lower=0, upper=1> fit_cQ;
  real<lower=0> c0_fixed;          // counts/keV; used only if fit_c0 = 0
  real<lower=0> cQ_fixed;          // counts/keV; used only if fit_cQ = 0
  // Lognormal prior locations/scales, in the parameter order below.
  vector[4] log_rate_prior_mean;   // N_2nu, N_ar39, c0, cQ
  vector<lower=0>[4] log_rate_prior_sd;
  vector[5] log_shape_prior_mean;  // mu, sigma, tau, base_alpha, base_beta
  vector<lower=0>[5] log_shape_prior_sd;
}
transformed data {
  vector[n_bins + 1] pr_cdf;
  vector[n_bins] pr_bin_probability;
  vector[n_bins] linear_c0_integral;
  vector[n_bins] linear_cQ_integral;
  int n_fit = sum(use_bin);
  array[n_fit] int fit_idx;
  if (n_fit == 0) reject("At least one bin must have use_bin = 1");
  {
    int k = 1;
    for (b in 1:n_bins) {
      if (use_bin[b] == 1) {
        fit_idx[k] = b;
        k += 1;
      }
    }
  }
  if (Q <= 0 || scale <= 0)
    reject("Q and scale must be strictly positive");
  if (min(log_rate_prior_sd) <= 0 || min(log_shape_prior_sd) <= 0)
    reject("Prior standard deviations must be strictly positive");
  if (edges[n_bins + 1] > Q) reject("Bin edges must lie within [0, Q]");
  for (i in 1:(n_bins + 1))
    pr_cdf[i] = primakoff_rosen_cdf(edges[i] | Q);
  for (b in 1:n_bins) {
    real lo = edges[b];
    real hi = edges[b + 1];
    real width = hi - lo;
    real midpoint_fraction = (hi + lo) / (2 * Q);
    if (width <= 0) reject("Bin edges must be strictly increasing");
    pr_bin_probability[b] = pr_cdf[b + 1] - pr_cdf[b];
    linear_c0_integral[b] = width * (1 - midpoint_fraction);
    linear_cQ_integral[b] = width * midpoint_fraction;
  }
}
parameters {
  real<lower=0> N_2nu;             // counts integrated over [0, Q]
  real<lower=0> N_ar39;            // counts over [loc, loc + scale]
  // Fixable parameters are arrays whose length is the data flag: length 1
  // when sampled, length 0 when fixed. The scalar used by the model is
  // defined in transformed parameters. Any other parameter can be made
  // fixable the same way: add fit_<name> and <name>_fixed to data, declare
  // array[fit_<name>] real<lower=0> <name>_free here, define <name> in
  // transformed parameters, and give the prior to <name>_free.
  array[fit_c0] real<lower=0> c0_free;  // counts/keV at E = 0
  array[fit_cQ] real<lower=0> cQ_free;  // counts/keV at E = Q
  real<lower=0> mu;
  real<lower=0> sigma;             // beta shape, not an energy resolution
  real<lower=0> tau;
  real<lower=0> base_alpha;
  real<lower=0> base_beta;
}
transformed parameters {
  real<lower=0> c0 = fit_c0 ? c0_free[1] : c0_fixed;
  real<lower=0> cQ = fit_cQ ? cQ_free[1] : cQ_fixed;
  vector[n_bins + 1] ar39_cdf;
  vector[n_bins] ar39_bin_probability;
  vector[n_bins] expected_2nu = N_2nu * pr_bin_probability;
  vector[n_bins] expected_linear = c0 * linear_c0_integral
                                   + cQ * linear_cQ_integral;
  vector[n_bins] expected_ar39;
  vector[n_bins] expected_counts;
  for (i in 1:(n_bins + 1))
    ar39_cdf[i] = modified_beta_g_cdf(edges[i] | mu, sigma, tau,
                                    base_alpha, base_beta, loc, scale);
  for (b in 1:n_bins)
    ar39_bin_probability[b] = ar39_cdf[b + 1] - ar39_cdf[b];
  expected_ar39 = N_ar39 * ar39_bin_probability;
  expected_counts = expected_2nu + expected_linear + expected_ar39;
}
model {
  // target += with _lpdf/_lpmf keeps every normalizing constant, so lp__ is
  // the full log posterior density (needed for marginal likelihoods). A prior
  // on a length-0 array adds nothing.
  target += lognormal_lpdf(N_2nu | log_rate_prior_mean[1], log_rate_prior_sd[1]);
  target += lognormal_lpdf(N_ar39 | log_rate_prior_mean[2], log_rate_prior_sd[2]);
  target += lognormal_lpdf(c0_free | log_rate_prior_mean[3], log_rate_prior_sd[3]);
  target += lognormal_lpdf(cQ_free | log_rate_prior_mean[4], log_rate_prior_sd[4]);
  target += lognormal_lpdf(mu | log_shape_prior_mean[1], log_shape_prior_sd[1]);
  target += lognormal_lpdf(sigma | log_shape_prior_mean[2], log_shape_prior_sd[2]);
  target += lognormal_lpdf(tau | log_shape_prior_mean[3], log_shape_prior_sd[3]);
  target += lognormal_lpdf(base_alpha | log_shape_prior_mean[4],
                           log_shape_prior_sd[4]);
  target += lognormal_lpdf(base_beta | log_shape_prior_mean[5],
                           log_shape_prior_sd[5]);
  // Masked bins are predicted but not compared with data.
  target += poisson_lpmf(counts[fit_idx] | expected_counts[fit_idx]);
}
generated quantities {
  // log_lik covers fitted bins only (in fit_idx order); counts_rep covers
  // every bin, so predictions are available inside excluded windows too.
  vector[n_fit] log_lik;
  array[n_bins] int counts_rep;
  for (k in 1:n_fit)
    log_lik[k] = poisson_lpmf(counts[fit_idx[k]] | expected_counts[fit_idx[k]]);
  for (b in 1:n_bins)
    counts_rep[b] = poisson_rng(expected_counts[b]);
}
