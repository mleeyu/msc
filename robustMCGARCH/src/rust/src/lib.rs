#![allow(non_snake_case)]

use extendr_api::prelude::*;
use statrs::distribution::{Continuous, Normal, StudentsT};
use statrs::function::gamma::ln_gamma;
use statrs::statistics::Statistics;
use nlopt::{Algorithm, Nlopt, Target};
use rayon::prelude::*;




enum Distribution {
    Normal,
    StudentsT,
}




#[extendr]
struct GARCH {
    white_noise: Distribution,
}

/// GARCH(p = 1, q = 1) model.
/// @export
#[extendr]
impl GARCH {
    /// Create GARCH(1,1) model with specified white noise distribution.
    fn new(white_noise: &str) -> Self {
        Self {
            white_noise: match white_noise {
                "Normal" => Distribution::Normal,
                "StudentsT" => Distribution::StudentsT,
                _ => panic!("Unknown distribution: {}", white_noise),
            },
        }
    }

    /// Simulate returns and sigmas from GARCH(p = 1, q = 1) model.
    fn simulate(&self, params: &[f64], n: usize) -> List {
        let (omega, alpha, beta): (f64, f64, f64) =
            (params[0], params[1], params[2]);
        let mut returns: Vec<f64> = vec![0.0_f64; n];
        let mut sigmas: Vec<f64> = vec![0.0_f64; n];
        let errors: Vec<f64> = match &self.white_noise {
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

        match &self.white_noise {
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

        match &self.white_noise {
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

    fn objective_p(&self, params: &[f64], returns: &[f64]) -> f64 {
        let (omega, alpha, beta): (f64, f64, f64) = (params[0], params[1], params[2]);
        if alpha + beta >= 1.0_f64 { return f64::NEG_INFINITY; }

        match &self.white_noise {
            Distribution::Normal => {
                let sigma2_unconditional: f64 = omega / (1.0_f64 - alpha - beta);
                let chunk_size: usize = returns.len().div_ceil(rayon::current_num_threads());
                let objective: f64 =
                    - returns[0] * returns[0] / sigma2_unconditional - sigma2_unconditional.ln()
                    + &returns[1..]
                    .par_chunks(chunk_size)
                    .enumerate()
                    .map(|(chunk_idx, chunk)| {
                        // Non recursive
                        let mut sigma2: f64 = 0.0_f64;
                        let i_chunk: usize = 1_usize + chunk_idx * chunk_size;
                        let mut beta_power: f64 = 1.0_f64;
                        for &r in returns[0..i_chunk].iter().rev() {
                            if beta_power < 1e-300 { break; }
                            sigma2 += beta_power * (omega + alpha * r * r);
                            beta_power *= beta;
                        }
                        sigma2 += beta_power * sigma2_unconditional;

                        // Recursive
                        let mut objective_chunk: f64 = - chunk[0] * chunk[0] / sigma2 - sigma2.ln();
                        for i in 1..chunk.len() {
                            let prev: f64 = chunk[i - 1];
                            let curr: f64 = chunk[i];
                            sigma2 = omega + alpha * prev * prev + beta * sigma2;
                            objective_chunk += - curr * curr / sigma2 - sigma2.ln();
                        }
                        objective_chunk
                    })
                    .sum::<f64>();
                objective
            }
            Distribution::StudentsT => {
                let mut sigma2 = omega / (1.0_f64 - alpha - beta);
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
                        |x: &[f64]| self.objective_p(x, data),
                        grad,
                    );
                }

                self.objective_p(x, data)
            },
            Target::Maximize,
            returns,
        );

        let (lower_bounds, upper_bounds) = match &self.white_noise {
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
    white_noise: Distribution,
}

/// MCGARCH model.
/// @export
#[extendr]
impl MCGARCH {
    fn new(white_noise: &str) -> Self {
        Self {
            white_noise: match white_noise {
                "Normal" => Distribution::Normal,
                "StudentsT" => Distribution::StudentsT,
                _ => panic!("Unknown distribution: {}", white_noise),
            },
        }
    }

    fn fit(&self, returns_ti: &[f64], n_bins: usize) -> List {
        let n_days: usize = returns_ti.len() / n_bins;
        let mut returns_t: Vec<f64> = vec![0.0_f64; n_days];
        for (t, c) in returns_ti.chunks_exact(n_bins).enumerate() {
            returns_t[t] = c.iter().sum();
        }
        let (garch_t, initial_params_t): (GARCH, Vec<f64>) =
            match &self.white_noise {
                Distribution::Normal => {
                    (
                        GARCH::new("Normal"),
                        vec![
                            returns_t.clone().variance() / 1000_f64,
                            0.05_f64,
                            0.9_f64
                        ]
                    )
                },
                Distribution::StudentsT => {
                    (
                        GARCH::new("StudentsT"),
                        vec![
                            returns_t.clone().variance() / 1000_f64,
                            0.05_f64,
                            0.9_f64,
                            4.0_f64
                        ]
                    )
                },
            };
        let params_t: Vec<f64> = garch_t.fit(&initial_params_t, &returns_t);
        let sigmas_t = garch_t.sigmas(
            &params_t,
            &returns_t,
        );

        let mut sigmas_i = vec![0.0_f64; n_bins];
        for (t, c) in returns_ti.chunks_exact(n_bins).enumerate() {
            for i in 0..n_bins {
                sigmas_i[i] += (c[i] / sigmas_t[t]).powi(2);
            }
        }
        for i in 0..n_bins {
            sigmas_i[i] = (sigmas_i[i] / n_days as f64).sqrt();
        }

        let mut returns_ti_normalized: Vec<f64> = vec![0.0_f64; n_days * n_bins];
        for (t, (c1, c2)) in returns_ti_normalized
            .chunks_exact_mut(n_bins)
            .zip(returns_ti.chunks_exact(n_bins))
            .enumerate()
        {
            for i in 0..n_bins {
                c1[i] = c2[i] / (sigmas_t[t] * sigmas_i[i]);
            }
        }
        let garch_ti: GARCH = GARCH::new("Normal");
        let params_ti: Vec<f64> = garch_ti.fit(
            &vec![
                returns_ti_normalized.clone().population_variance() / 1000_f64,
                0.05,
                0.9
            ],
            &returns_ti_normalized
        );
        let sigmas_ti: Vec<f64> = garch_ti.sigmas(
            &params_ti,
            &returns_ti_normalized,
        );

        list!(
            params_t = params_t,
            sigmas_t = sigmas_t,
            sigmas_i = sigmas_i,
            params_ti = params_ti,
            sigmas_ti = sigmas_ti
        )
    }
}




// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod robustMCGARCH;
    impl GARCH;
    impl MCGARCH;
}
