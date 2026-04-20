use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// A single uncertainty scenario for robust optimization.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub id: usize,
    pub wind_profile: Vec<(f64, Vector3<f64>)>,
    pub density_factor: f64,
    pub cd_factor: f64,
    pub thrust_bias: f64,
}

impl Scenario {
    /// Nominal (no-perturbation) scenario.
    pub fn nominal(id: usize) -> Self {
        Self {
            id,
            wind_profile: vec![(0.0, Vector3::zeros()), (100_000.0, Vector3::zeros())],
            density_factor: 1.0,
            cd_factor: 1.0,
            thrust_bias: 0.0,
        }
    }
}

/// Configuration for scenario generation.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    /// Standard deviation of wind speed at each altitude layer (m/s).
    pub wind_sigma: f64,
    /// Standard deviation of atmospheric density perturbation (fraction).
    pub density_sigma: f64,
    /// Standard deviation of Cd perturbation (fraction).
    pub cd_sigma: f64,
    /// Standard deviation of thrust bias (N).
    pub thrust_bias_sigma: f64,
    /// Altitude breakpoints for wind profile generation.
    pub wind_altitudes: Vec<f64>,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            wind_sigma: 10.0,
            density_sigma: 0.05,
            cd_sigma: 0.08,
            thrust_bias_sigma: 50.0,
            wind_altitudes: vec![0.0, 1000.0, 5000.0, 10_000.0, 20_000.0, 50_000.0, 80_000.0],
        }
    }
}

/// Generate a set of random scenarios for Monte Carlo evaluation.
pub fn generate_scenarios(n: usize, seed: u64, config: &ScenarioConfig) -> Vec<Scenario> {
    let mut rng = StdRng::seed_from_u64(seed);
    let wind_dist = Normal::new(0.0, config.wind_sigma).unwrap();
    let density_dist = Normal::new(1.0, config.density_sigma).unwrap();
    let cd_dist = Normal::new(1.0, config.cd_sigma).unwrap();
    let thrust_dist = Normal::new(0.0, config.thrust_bias_sigma).unwrap();

    (0..n)
        .map(|id| {
            let wind_profile: Vec<(f64, Vector3<f64>)> = config
                .wind_altitudes
                .iter()
                .map(|&alt| {
                    let wx = wind_dist.sample(&mut rng);
                    let wy = wind_dist.sample(&mut rng);
                    (alt, Vector3::new(wx, wy, 0.0))
                })
                .collect();

            let density_factor = density_dist.sample(&mut rng).max(0.5);
            let cd_factor = cd_dist.sample(&mut rng).max(0.5);
            let thrust_bias = thrust_dist.sample(&mut rng);

            Scenario {
                id,
                wind_profile,
                density_factor,
                cd_factor,
                thrust_bias,
            }
        })
        .collect()
}
