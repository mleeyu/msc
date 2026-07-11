use extendr_api::prelude::*;

/// Return string `"Hello world!"` to R.
/// @export
#[extendr]
fn hello_world() -> &'static str {
    "Hello world!"
}




enum Distribution {
    Normal,
    StudentT,
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
                "StudentT" => Distribution::StudentT,
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
            Distribution::StudentT => {
                let nu: f64 = params[3];
                let inv_sd: f64 = 1.0_f64 / (nu / (nu - 2.0_f64)).sqrt();
                call!("rt", n, nu)
                    .unwrap()
                    .as_real_vector()
                    .unwrap()
                    .into_iter()
                    .map(|x| x * inv_sd)
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
}




// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod robustMCGARCH;
    fn hello_world;

    impl GARCH;
}
