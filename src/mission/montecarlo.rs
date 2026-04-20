/// Monte Carlo evaluation framework.
///
/// Runs many forward simulations with randomized scenario parameters
/// to evaluate trajectory robustness and collect landing statistics.

use crate::aero::AeroTable;
use crate::dynamics::State;
use crate::mission::simulate::{
    Controller, SimulationConfig, SimulationResult, forward_simulate,
};
use crate::robust::{Scenario, ScenarioConfig, generate_scenarios};
use crate::vehicle::VehicleParams;
use rayon::prelude::*;

/// Configuration for a Monte Carlo campaign.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    pub n_samples: usize,
    pub seed: u64,
    pub parallel: bool,
    pub scenario_config: ScenarioConfig,
    pub sim_config: SimulationConfig,
    pub success_landing_error: f64,
    pub success_landing_speed: f64,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            seed: 42,
            parallel: true,
            scenario_config: ScenarioConfig::default(),
            sim_config: SimulationConfig::default(),
            success_landing_error: 20.0,
            success_landing_speed: 5.0,
        }
    }
}

/// Aggregate statistics from a Monte Carlo campaign.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    pub n_total: usize,
    pub n_landed: usize,
    pub n_success: usize,
    pub n_aborted: usize,
    pub success_rate: f64,
    pub landing_error_mean: f64,
    pub landing_error_max: f64,
    pub final_speed_mean: f64,
    pub final_speed_max: f64,
    pub fuel_used_mean: f64,
    pub fuel_used_max: f64,
    pub max_g_load_mean: f64,
    pub max_g_load_max: f64,
    pub max_q_dyn_mean: f64,
    pub max_q_dyn_max: f64,
    pub individual_results: Vec<(usize, SimulationResult)>,
}

impl MonteCarloResult {
    pub fn summary(&self) -> String {
        format!(
            "Monte Carlo: {}/{} success ({:.1}%)\n\
             Landing error: mean={:.1}m, max={:.1}m\n\
             Final speed:   mean={:.1}m/s, max={:.1}m/s\n\
             Fuel used:     mean={:.1}kg, max={:.1}kg\n\
             Max G-load:    mean={:.1}g, max={:.1}g\n\
             Max Q:         mean={:.0}Pa, max={:.0}Pa\n\
             Aborted:       {}/{}",
            self.n_success, self.n_total, self.success_rate * 100.0,
            self.landing_error_mean, self.landing_error_max,
            self.final_speed_mean, self.final_speed_max,
            self.fuel_used_mean, self.fuel_used_max,
            self.max_g_load_mean, self.max_g_load_max,
            self.max_q_dyn_mean, self.max_q_dyn_max,
            self.n_aborted, self.n_total,
        )
    }
}

/// Run a Monte Carlo campaign.
///
/// Generates scenarios, runs forward simulations for each, and collects statistics.
pub fn run_montecarlo<C: Controller + Sync>(
    initial_state: &State,
    vehicle: &VehicleParams,
    aero: &AeroTable,
    controller: &C,
    config: &MonteCarloConfig,
) -> MonteCarloResult {
    let scenarios = generate_scenarios(
        config.n_samples,
        config.seed,
        &config.scenario_config,
    );

    let run_one = |scenario: &Scenario| -> (usize, SimulationResult) {
        let result = forward_simulate(
            initial_state,
            vehicle,
            aero,
            scenario,
            controller,
            &config.sim_config,
        );
        (scenario.id, result)
    };

    let results: Vec<(usize, SimulationResult)> = if config.parallel {
        scenarios.par_iter().map(run_one).collect()
    } else {
        scenarios.iter().map(run_one).collect()
    };

    compute_statistics(results, config)
}

/// Run Monte Carlo with pre-generated scenarios.
pub fn run_montecarlo_with_scenarios<C: Controller + Sync>(
    initial_state: &State,
    vehicle: &VehicleParams,
    aero: &AeroTable,
    controller: &C,
    scenarios: &[Scenario],
    config: &MonteCarloConfig,
) -> MonteCarloResult {
    let run_one = |scenario: &Scenario| -> (usize, SimulationResult) {
        let result = forward_simulate(
            initial_state,
            vehicle,
            aero,
            scenario,
            controller,
            &config.sim_config,
        );
        (scenario.id, result)
    };

    let results: Vec<(usize, SimulationResult)> = if config.parallel {
        scenarios.par_iter().map(run_one).collect()
    } else {
        scenarios.iter().map(run_one).collect()
    };

    compute_statistics(results, config)
}

fn compute_statistics(
    results: Vec<(usize, SimulationResult)>,
    config: &MonteCarloConfig,
) -> MonteCarloResult {
    let n_total = results.len();

    let n_landed = results.iter().filter(|(_, r)| r.landed).count();
    let n_success = results.iter()
        .filter(|(_, r)| r.is_success(config.success_landing_error, config.success_landing_speed))
        .count();
    let n_aborted = results.iter().filter(|(_, r)| r.aborted).count();

    let landing_errors: Vec<f64> = results.iter().map(|(_, r)| r.landing_error).collect();
    let final_speeds: Vec<f64> = results.iter().map(|(_, r)| r.final_speed).collect();
    let fuels: Vec<f64> = results.iter().map(|(_, r)| r.fuel_used).collect();
    let g_loads: Vec<f64> = results.iter().map(|(_, r)| r.max_g_load).collect();
    let q_dyns: Vec<f64> = results.iter().map(|(_, r)| r.max_dynamic_pressure).collect();

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };
    let max = |v: &[f64]| -> f64 {
        v.iter().cloned().fold(0.0_f64, f64::max)
    };

    MonteCarloResult {
        n_total,
        n_landed,
        n_success,
        n_aborted,
        success_rate: if n_total > 0 { n_success as f64 / n_total as f64 } else { 0.0 },
        landing_error_mean: mean(&landing_errors),
        landing_error_max: max(&landing_errors),
        final_speed_mean: mean(&final_speeds),
        final_speed_max: max(&final_speeds),
        fuel_used_mean: mean(&fuels),
        fuel_used_max: max(&fuels),
        max_g_load_mean: mean(&g_loads),
        max_g_load_max: max(&g_loads),
        max_q_dyn_mean: mean(&q_dyns),
        max_q_dyn_max: max(&q_dyns),
        individual_results: results,
    }
}
