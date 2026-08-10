devtools::load_all("robustMCGARCH")


set.seed(1)
n_days <- 2500
n_bins <- 288
n_burnin <- 1000


# Simulate
garch_studentst <- GARCH$new("StudentsT")
# daily var (h_t)
h_t <- garch_studentst$simulate(c(0.01, 0.1, 0.85, 7.0), n_burnin + n_days)$sigmas^2
h_t <- tail(h_t, n_days)
# diurnal var (s_i), with u shape
s_i <- (1 + 6 * seq(-1, 1, length.out = n_bins)^2) * abs(rnorm(n_bins, sd = 0.01))
s_i <- s_i / mean(s_i)
# intraday var (q_ti)
garch_normal <- GARCH$new("Normal")
q_ti <- garch_normal$simulate(c(0.01, 0.1, 0.85), n_burnin + n_days * n_bins)$sigmas^2
q_ti <- tail(q_ti, n_days * n_bins)
# intraday returns
errors <- rnorm(n_days * n_bins)
returns_ti <- rep(0, n_days * n_bins)
for (t in 1:n_days) {
  for (i in 1:n_bins) {
    index <- (t - 1) * n_bins + i
    returns_ti[index] <- sqrt(h_t[t] * s_i[i] * q_ti[index]) * errors[index]
  }
}


# Estimate
# daily returns
returns_t <- rep(0, n_days)
for (t in 1:n_days) {
  indexes <- (t - 1) * n_bins + 1:n_bins
  returns_t[t] <- sum(returns_ti[indexes])
}
# daily var (h_t_hat)
garch_studentst <- GARCH$new("StudentsT")
params_t <- garch_studentst$fit(c(var(returns_t) / 1000, 0.05, 0.9, 4.0), returns_t)
params_t
h_t_hat <- garch_studentst$sigmas(params_t, returns_t)^2
# diurnal var (s_i_hat)
y_ti <- returns_ti / rep(sqrt(h_t_hat), each = n_bins)
s_i_hat <- rep(0, n_bins)
for (i in 1:n_bins) {
  indexes <- (1:n_days - 1) * n_bins + i
  s_i_hat[i] <- mean(y_ti[indexes]^2)
}
z_ti <- returns_ti / sqrt(rep(h_t_hat, each = n_bins) * rep(s_i_hat, times = n_days))
# intraday var (q_ti_hat)
garch_normal <- GARCH$new("Normal")
params_ti <- garch_normal$fit(c(var(z_ti) / 1000, 0.05, 0.9), z_ti)
params_ti
q_ti_hat <- garch_normal$sigmas(params_ti, z_ti)^2
