#![allow(non_snake_case)]

use extendr_api::prelude::*;
use statrs::distribution::{Continuous, Normal, StudentsT};
use statrs::function::gamma::ln_gamma;
use statrs::statistics::Statistics;
use nlopt::{Algorithm, Nlopt, Target};

/// Return string `"Hello world!"` to R.
/// @export
#[extendr]
fn hello_world() -> &'static str {
    "Hello world!"
}




enum Distribution {
    Normal,
    StudentsT,
}

#[extendr]
struct GARCH {
    distribution: Distribution,
}

/// GARCH(p = 1, q = 1) model.
/// @export
#[extendr]
impl GARCH {
    /// Create GARCH(1,1) model with specified error distribution.
    fn new(distribution: &str) -> Self {
        Self {
            distribution: match distribution {
                "Normal" => Distribution::Normal,
                "StudentsT" => Distribution::StudentsT,
                _ => panic!("Unknown distribution: {}", distribution),
            },
        }
    }

    /// Simulate returns and sigmas from GARCH(p = 1, q = 1) model.
    fn simulate(&self, params: &[f64], n: usize) -> List {
        let (omega, alpha, beta): (f64, f64, f64) =
            (params[0], params[1], params[2]);
        let mut returns: Vec<f64> = vec![0.0_f64; n];
        let mut sigmas: Vec<f64> = vec![0.0_f64; n];
        let errors: Vec<f64> = match &self.distribution {
            Distribution::Normal => {
                call!("rnorm", n, 0.0_f64, 1.0_f64)
                    .unwrap()
                    .as_real_vector()
                    .unwrap()
            }
            Distribution::StudentsT => {
                let nu: f64 = params[3];
                let inv_sd: f64 = 1.0_f64 / (nu / (nu - 2.0_f64)).sqrt();
                call!("rt", n, nu)
                    .unwrap()
                    .as_real_vector()
                    .unwrap()
                    .into_iter()
                    .map(|x: f64| x * inv_sd)
                    .collect::<Vec<f64>>()
            }
        };

        sigmas[0] = (omega / (1.0_f64 - alpha - beta)).sqrt();
        for i in 1..n {
            sigmas[i] = (omega
                        + alpha * returns[i - 1].powi(2)
                        + beta * sigmas[i - 1].powi(2)
                        ).sqrt();
            returns[i] = sigmas[i] * errors[i];
        }

        list!(returns = returns, sigmas = sigmas)
    }

    /// Calculate sigmas of GARCH(p = 1, q = 1) model.
    fn sigmas(&self, params: &[f64], returns: &[f64]) -> Vec<f64> {
        let (omega, alpha, beta): (f64, f64, f64) =
            (params[0], params[1], params[2]);
        let n: usize = returns.len();
        let mut sigmas: Vec<f64> = vec![0.0_f64; n];

        sigmas[0] = (omega / (1.0_f64 - alpha - beta)).sqrt();
        for i in 1..n {
            sigmas[i] = (omega
                         + alpha * returns[i - 1].powi(2)
                         + beta * sigmas[i - 1].powi(2)
                        ).sqrt();
        }

        sigmas
    }

    /// Forecast one-step-ahead sigma of GARCH(p = 1, q = 1) model.
    fn forecast(&self, params: &[f64], returns: &[f64]) -> f64 {
        let (omega, alpha, beta): (f64, f64, f64) =
            (params[0], params[1], params[2]);
        let n: usize = returns.len();
        let mut sigma: f64 = (omega / (1.0_f64 - alpha - beta)).sqrt();

        for i in 1..(n + 1) {
            sigma = (omega
                     + alpha * returns[i - 1].powi(2)
                     + beta * sigma.powi(2)
                    ).sqrt();
        }

        sigma
    }

    /// Log-likelihood function of GARCH(p = 1, q = 1) model.
    fn log_likelihood(&self, params: &[f64], returns: &[f64]) -> f64 {
        let (omega, alpha, beta): (f64, f64, f64) =
            (params[0], params[1], params[2]);
        let mut sigma: f64 = (omega / (1.0_f64 - alpha - beta)).sqrt();

        match &self.distribution {
            Distribution::Normal => {
                let normal = Normal::new(0.0_f64, 1.0_f64).unwrap();
                let mut log_likelihood: f64 =
                    normal.ln_pdf(returns[0] / sigma) - sigma.ln();
                for w in returns.windows(2) {
                    sigma = (omega
                             + alpha * w[0].powi(2)
                             + beta * sigma.powi(2)
                            ).sqrt();
                    log_likelihood +=
                        normal.ln_pdf(w[1] / sigma) - sigma.ln();
                }
                log_likelihood
            }
            Distribution::StudentsT => {
                let nu: f64 = params[3];
                let inv_sd: f64 = 1.0_f64 / (nu / (nu - 2.0_f64)).sqrt();
                let studentst = StudentsT::new(0.0_f64, 1.0_f64, nu).unwrap();
                let mut log_likelihood: f64 =
                    studentst.ln_pdf(returns[0] / (sigma * inv_sd))
                    - (sigma * inv_sd).ln();
                for w in returns.windows(2) {
                    sigma = (omega
                             + alpha * w[0].powi(2)
                             + beta * sigma.powi(2)
                            ).sqrt();
                    log_likelihood +=
                        studentst.ln_pdf(w[1] / (sigma * inv_sd))
                        - (sigma * inv_sd).ln();
                }
                log_likelihood
            }
        }
    }

    /// Objective function of GARCH(p = 1, q = 1) model.
    fn objective(&self, params: &[f64], returns: &[f64]) -> f64 {
        let (omega, alpha, beta): (f64, f64, f64) =
            (params[0], params[1], params[2]);
        if alpha + beta >= 1.0_f64 {
            return f64::NEG_INFINITY;
        }
        let mut sigma2: f64 = omega / (1.0_f64 - alpha - beta);

        match &self.distribution {
            Distribution::Normal => {
                let mut objective: f64 =
                    - returns[0] * returns[0] / sigma2 - sigma2.ln();
                for w in returns.windows(2) {
                    sigma2 = omega
                             + alpha * w[0] * w[0]
                             + beta * sigma2;
                    objective +=
                        - w[1] * w[1] / sigma2 - sigma2.ln();
                }
                objective
            }
            Distribution::StudentsT => {
                let nu: f64 = params[3];
                let nu_plus_1: f64 = nu + 1.0_f64;
                let inv_sd2: f64 = 1.0_f64 / (nu / (nu - 2.0_f64));
                let mut objective: f64 =
                    2.0_f64 * (ln_gamma(nu_plus_1 / 2.0_f64)
                               - ln_gamma(nu / 2.0_f64)) - nu.ln()
                    - nu_plus_1 * (1.0_f64 + returns[0] * returns[0]
                                             / (sigma2 * nu)).ln()
                    - sigma2.ln();
                for w in returns.windows(2) {
                    sigma2 = omega
                             + alpha * w[0] * w[0]
                             + beta * sigma2;
                    sigma2 *= inv_sd2;
                    objective +=
                        2.0_f64 * (ln_gamma(nu_plus_1 / 2.0_f64)
                                   - ln_gamma(nu / 2.0_f64)) - nu.ln()
                        - nu_plus_1 * (1.0_f64 + w[1] * w[1]
                                                 / (sigma2 * nu)).ln()
                        - sigma2.ln();
                }
                objective
            }
        }
    }

    // Fit returns with GARCH(p = 1, q = 1) model.
    fn fit(&self, params: &[f64], returns: &[f64]) -> Vec<f64> {
        let n = params.len();

        let mut opt = Nlopt::new(
            Algorithm::Lbfgs,
            n,
            |x: &[f64], grad: Option<&mut [f64]>, data: &mut &[f64]| -> f64 {
                if let Some(grad) = grad {
                    nlopt::approximate_gradient(
                        x,
                        |x: &[f64]| self.objective(x, data),
                        grad,
                    );
                }

                self.objective(x, data)
            },
            Target::Maximize,
            returns,
        );

        let (lower_bounds, upper_bounds) = match &self.distribution {
            Distribution::Normal => (
                vec![f64::EPSILON.sqrt(),
                     0.0_f64,
                     0.0_f64
                ],
                vec![f64::INFINITY,
                     1.0_f64 - f64::EPSILON.sqrt(),
                     1.0_f64 - f64::EPSILON.sqrt()
                ],
            ),
            Distribution::StudentsT => (
                vec![f64::EPSILON.sqrt(),
                     0.0_f64,
                     0.0_f64,
                     2.0_f64 + f64::EPSILON.sqrt()
                ],
                vec![f64::INFINITY,
                     1.0_f64 - f64::EPSILON.sqrt(),
                     1.0_f64 - f64::EPSILON.sqrt(),
                     100.0_f64
                ],
            ),
        };
        opt.set_lower_bounds(&lower_bounds).unwrap();
        opt.set_upper_bounds(&upper_bounds).unwrap();

        opt.set_xtol_rel(2.2e-9).unwrap();
        opt.set_ftol_rel(2.2e-9).unwrap();
        opt.set_maxeval(1000).unwrap();

        let mut solution = params.to_vec();
        match opt.optimize(&mut solution) {
            Ok(_) => solution,
            Err(_) => params.to_vec(),
        }
    }
}

#[extendr]
struct MCGARCH {
    distribution: Distribution,
}

/// MCGARCH model.
/// @export
#[extendr]
impl MCGARCH {
    fn new(distribution: &str) -> Self {
        Self {
            distribution: match distribution {
                "Normal" => Distribution::Normal,
                "StudentsT" => Distribution::StudentsT,
                _ => panic!("Unknown distribution: {}", distribution),
            },
        }
    }

    fn fit(&self, intraday_returns: &[f64], n_bins: usize) -> Vec<f64> {
        let n_days: usize = intraday_returns.len() / n_bins;
        let mut daily_returns: Vec<f64> = vec![0.0_f64; n_days];
        for (i, c) in intraday_returns.chunks_exact(n_bins).enumerate() {
            daily_returns[i] = c.iter().sum();
        }
        let (daily_garch, daily_init_params): (GARCH, Vec<f64>) =
            match &self.distribution {
                Distribution::Normal => {
                    (
                        GARCH::new("Normal"),
                        vec![
                            daily_returns.clone().variance() / 1000_f64,
                            0.05_f64,
                            0.9_f64
                        ]
                    )
                },
                Distribution::StudentsT => {
                    (
                        GARCH::new("StudentsT"),
                        vec![
                            daily_returns.clone().variance() / 1000_f64,
                            0.05_f64,
                            0.9_f64,
                            4.0_f64
                        ]
                    )
                },
            };
        let daily_sigmas = daily_garch.sigmas(
            &daily_garch.fit(&daily_init_params, &daily_returns),
            &daily_returns,
        );

        let mut diurnal_sigmas2 = vec![0.0_f64; n_bins];
        for (i, c) in intraday_returns.chunks_exact(n_bins).enumerate() {
            for j in 1..n_bins {
                diurnal_sigmas2[j] += c[j].powi(2) / daily_sigmas[i];
            }
        }
        for i in 1..n_bins {
            diurnal_sigmas2[i] /= n_days as f64;
        }

        let mut normalized_intraday_returns: Vec<f64> = vec![0.0_f64; n_days * n_bins];
        for (i, (c1, c2)) in normalized_intraday_returns.chunks_exact_mut(n_bins)
            .zip(intraday_returns.chunks_exact(n_bins))
            .enumerate() {
            for j in 1..n_bins {
                c1[j] = c2[j] / (daily_sigmas[i] * diurnal_sigmas2[j].sqrt());
            }
        }
        let intraday_garch: GARCH = GARCH::new("Normal");
        intraday_garch.fit(
            &vec![
                normalized_intraday_returns.clone().variance() / 1000_f64,
                0.05,
                0.9
            ],
            &normalized_intraday_returns
        )
    }
}




// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod robustMCGARCH;
    fn hello_world;

    impl GARCH;
    impl MCGARCH;
}
