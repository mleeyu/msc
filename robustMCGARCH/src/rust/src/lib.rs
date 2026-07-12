use extendr_api::prelude::*;
use statrs::distribution::{Normal, StudentsT, Continuous};

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

#[derive(IntoList)]
struct SimulateData {
    returns: Vec<f64>,
    sigmas: Vec<f64>,
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
    fn simulate(&self, params: &[f64], n: usize) -> SimulateData {
        let [omega, alpha, beta]: [f64; 3] = [params[0], params[1], params[2]];
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

        SimulateData { returns, sigmas }
    }

    /// Forecast sigmas from GARCH(p = 1, q = 1) model.
    fn forecast(&self, params: &[f64], returns: &[f64]) -> Vec<f64> {
        let [omega, alpha, beta]: [f64; 3] = [params[0], params[1], params[2]];
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

    fn log_likelihood(&self, params: &[f64], returns: &[f64]) -> f64 {
        let sigmas: Vec<f64> = self.forecast(params, returns);

        match &self.distribution {
            Distribution::Normal => {
                let normal = Normal::new(0.0_f64, 1.0_f64).unwrap();
                returns
                    .iter()
                    .zip(sigmas.iter())
                    .map(|(&r, &s)| {
                        normal.ln_pdf(r / s) - s.ln()
                    })
                    .sum()
            }
            Distribution::StudentsT => {
                let nu: f64 = params[3];
                let inv_sd: f64 = 1.0_f64 / (nu / (nu - 2.0_f64)).sqrt();
                let studentst = StudentsT::new(0.0_f64, 1.0_f64, nu).unwrap();
                returns
                    .iter()
                    .zip(sigmas.iter())
                    .map(|(&r, &s)| {
                        studentst.ln_pdf(r / (s * inv_sd)) - (s * inv_sd).ln()
                    })
                    .sum()
            }
        }
    }
}




// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod robustMCGARCH;
    fn hello_world;

    impl GARCH;
}
