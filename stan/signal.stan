// Copyright (C) 2026 Francesco Borra
//
// Fixed-background signal fit for one DM mass hypothesis.
//
// For detector group g and energy bin i,
//   counts[g, i] ~ Poisson(b_hat[g, i] + R * signal_template[g, i]),
// with the product running only over bins with use_bin = 1. b_hat is the fixed
// background prediction from the broad background fit, in counts per bin.
// signal_template is the accepted signal exposure per bin, in kg yr, so R is
// the signal rate in counts / (kg yr), shared by all groups. The prior on R is
// uniform on [0, rate_upper].

data {
  int<lower=1> n_groups;
  int<lower=1> n_bins;
  vector[n_bins + 1] edges;
  array[n_groups, n_bins] int<lower=0> counts;
  array[n_groups, n_bins] int<lower=0, upper=1> use_bin;
  matrix<lower=0>[n_groups, n_bins] b_hat;
  matrix<lower=0>[n_groups, n_bins] signal_template;
  real<lower=0> rate_upper;
}

transformed data {
  int n_used = 0;
  for (i in 1:n_bins) {
    if (edges[i + 1] <= edges[i]) {
      reject("edges must increase strictly; found edges[", i + 1, "] = ",
             edges[i + 1], " <= edges[", i, "] = ", edges[i], ".");
    }
  }
  for (g in 1:n_groups) {
    for (i in 1:n_bins) {
      if (use_bin[g, i] == 1) {
        n_used += 1;
        if (b_hat[g, i] <= 0) {
          reject("b_hat must be positive in fitted bins; group ", g, ", bin ", i, ".");
        }
      }
    }
  }
  if (n_used == 0) {
    reject("At least one bin must enter the likelihood.");
  }
  if (rate_upper <= 0) {
    reject("rate_upper must be positive.");
  }
}

parameters {
  real<lower=0, upper=rate_upper> R;
}

model {
  // Uniform prior on [0, rate_upper] from the declared bounds.
  for (g in 1:n_groups) {
    for (i in 1:n_bins) {
      if (use_bin[g, i] == 1) {
        counts[g, i] ~ poisson(b_hat[g, i] + R * signal_template[g, i]);
      }
    }
  }
}

generated quantities {
  matrix[n_groups, n_bins] expected_signal = R * signal_template;
  matrix[n_groups, n_bins] expected_counts = b_hat + expected_signal;
  real total_signal = sum(expected_signal);
  array[n_groups, n_bins] int counts_rep;
  vector[n_used] log_lik;
  {
    int k = 1;
    for (g in 1:n_groups) {
      for (i in 1:n_bins) {
        counts_rep[g, i] = poisson_rng(expected_counts[g, i]);
        if (use_bin[g, i] == 1) {
          log_lik[k] = poisson_lpmf(counts[g, i] | expected_counts[g, i]);
          k += 1;
        }
      }
    }
  }
}
