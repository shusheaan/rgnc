/// Integration tests for the Monte Carlo campaign runner.
///
/// Validates that Monte Carlo evaluation produces meaningful statistics
/// across randomized scenarios.

use nalgebra::{UnitQuaternion, Vector3};
use rgnc::aero::AeroTable;
use rgnc::dynamics::State;
use rgnc::mission::montecarlo::*;
use rgnc::mission::simulate::*;
use rgnc::robust::{Scenario, ScenarioConfig, generate_scenarios};
use rgnc::vehicle::VehicleParams;

fn test_vehicle() -> VehicleParams {
    VehicleParams {
        dry_mass: 1505.0,
        fuel_mass: 300.0,
        thrust_max: 3100.0,
        thrust_min: 1400.0,
        isp: 225.0,
        ref_area: 1.0,
        glideslope_angle: 0.0698,
        inertia: Vector3::new(1000.0, 1000.0, 200.0),
        ref_length: 3.0,
        cg_offset: Vector3::zeros(),
        cp_offset: Vector3::new(2.0, 0.0, 0.0),
        engine_offset: Vector3::new(-3.0, 0.0, 0.0),
        gimbal_max: 0.2618,
    }
}

#[test]
fn test_scenario_generation_deterministic() {
    let config = ScenarioConfig::default();
    let scenarios1 = generate_scenarios(10, 42, &config);
    let scenarios2 = generate_scenarios(10, 42, &config);

    for (s1, s2) in scenarios1.iter().zip(scenarios2.iter()) {
        assert_eq!(s1.density_factor, s2.density_factor);
        assert_eq!(s1.cd_factor, s2.cd_factor);
        assert_eq!(s1.thrust_bias, s2.thrust_bias);
    }
}

#[test]
fn test_scenario_generation_different_seeds() {
    let config = ScenarioConfig::default();
    let scenarios1 = generate_scenarios(10, 42, &config);
    let scenarios2 = generate_scenarios(10, 99, &config);

    // Different seeds should produce different scenarios
    let different = scenarios1
        .iter()
        .zip(scenarios2.iter())
        .any(|(s1, s2)| s1.density_factor != s2.density_factor);

    assert!(different, "Different seeds should produce different scenarios");
}

#[test]
fn test_scenario_generation_physical_bounds() {
    let config = ScenarioConfig::default();
    let scenarios = generate_scenarios(100, 42, &config);

    for s in &scenarios {
        assert!(s.density_factor >= 0.5, "Density factor should be positive");
        assert!(s.cd_factor >= 0.5, "Cd factor should be positive");
        // Wind profile should have entries
        assert!(!s.wind_profile.is_empty(), "Wind profile should not be empty");
    }
}

#[test]
fn test_montecarlo_ballistic_freefall() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let controller = BallisticController;

    let config = MonteCarloConfig {
        n_samples: 20,
        seed: 42,
        parallel: false,
        scenario_config: ScenarioConfig {
            wind_sigma: 0.0,
            density_sigma: 0.0,
            cd_sigma: 0.0,
            thrust_bias_sigma: 0.0,
            ..Default::default()
        },
        sim_config: SimulationConfig {
            dt: 0.5,
            max_time: 50.0,
            ..Default::default()
        },
        success_landing_error: 100.0,
        success_landing_speed: 200.0,
    };

    let state = State::new(
        Vector3::new(0.0, 0.0, 1000.0),
        Vector3::new(0.0, 0.0, 0.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        vehicle.total_mass(),
        0.0,
    );

    let result = run_montecarlo(&state, &vehicle, &aero, &controller, &config);

    // All scenarios identical (zero perturbation) → all should land
    assert_eq!(result.n_total, 20);
    assert_eq!(result.n_landed, 20, "All should land: {}", result.summary());
    assert_eq!(result.n_aborted, 0, "None should abort");
}

#[test]
fn test_montecarlo_with_perturbations() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(1.2);
    let controller = BallisticController;

    let config = MonteCarloConfig {
        n_samples: 50,
        seed: 42,
        parallel: true,
        scenario_config: ScenarioConfig {
            wind_sigma: 10.0,
            density_sigma: 0.1,
            cd_sigma: 0.1,
            thrust_bias_sigma: 0.0,
            ..Default::default()
        },
        sim_config: SimulationConfig {
            dt: 0.5,
            max_time: 100.0,
            ..Default::default()
        },
        success_landing_error: 1000.0,
        success_landing_speed: 500.0,
    };

    let state = State::new(
        Vector3::new(0.0, 0.0, 5000.0),
        Vector3::new(50.0, 0.0, -100.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        vehicle.total_mass(),
        0.0,
    );

    let result = run_montecarlo(&state, &vehicle, &aero, &controller, &config);

    assert_eq!(result.n_total, 50);
    assert!(result.n_landed > 0, "At least some should land");

    // With perturbations, there should be spread in landing errors
    assert!(
        result.landing_error_max > result.landing_error_mean,
        "Max should exceed mean: max={:.1} mean={:.1}",
        result.landing_error_max,
        result.landing_error_mean
    );
}

#[test]
fn test_montecarlo_guided_vs_ballistic() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);

    let state = State::new(
        Vector3::new(100.0, 0.0, 2400.0),
        Vector3::new(0.0, 0.0, -75.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        vehicle.total_mass(),
        0.0,
    );

    let mc_config = MonteCarloConfig {
        n_samples: 20,
        seed: 42,
        parallel: false,
        scenario_config: ScenarioConfig {
            wind_sigma: 0.0,
            density_sigma: 0.0,
            cd_sigma: 0.0,
            thrust_bias_sigma: 0.0,
            ..Default::default()
        },
        sim_config: SimulationConfig {
            dt: 0.1,
            max_time: 200.0,
            ..Default::default()
        },
        success_landing_error: 50.0,
        success_landing_speed: 10.0,
    };

    // Ballistic: no guidance, should miss target
    let result_ballistic = run_montecarlo(
        &state,
        &vehicle,
        &aero,
        &BallisticController,
        &mc_config,
    );

    // Guided: gravity-turn, should do better
    let guided = GravityTurnController {
        target_pos: Vector3::new(0.0, 0.0, 0.0),
        vehicle: vehicle.clone(),
        gain_vel: 1.5,
        gain_pos: 0.3,
    };

    let result_guided = run_montecarlo(
        &state,
        &vehicle,
        &aero,
        &guided,
        &mc_config,
    );

    // Guided uses fuel (engine was on), ballistic uses none
    // This verifies the controller is actually commanding thrust
    assert!(
        result_guided.fuel_used_mean > result_ballistic.fuel_used_mean,
        "Guided controller should use fuel, ballistic should not.\n\
         Guided fuel: {:.1}kg, Ballistic fuel: {:.1}kg",
        result_guided.fuel_used_mean,
        result_ballistic.fuel_used_mean,
    );
}

#[test]
fn test_montecarlo_result_summary() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let controller = BallisticController;

    let config = MonteCarloConfig {
        n_samples: 5,
        seed: 42,
        parallel: false,
        sim_config: SimulationConfig {
            dt: 1.0,
            max_time: 20.0,
            ..Default::default()
        },
        ..Default::default()
    };

    let state = State::new(
        Vector3::new(0.0, 0.0, 500.0),
        Vector3::new(0.0, 0.0, 0.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        vehicle.total_mass(),
        0.0,
    );

    let result = run_montecarlo(&state, &vehicle, &aero, &controller, &config);

    let summary = result.summary();
    assert!(summary.contains("Monte Carlo"), "Summary should contain header");
    assert!(summary.contains("Landing error"), "Summary should contain landing error");
}

#[test]
fn test_montecarlo_with_scenarios() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let controller = BallisticController;

    let scenarios = vec![
        Scenario::nominal(0),
        Scenario::nominal(1),
        Scenario::nominal(2),
    ];

    let config = MonteCarloConfig {
        n_samples: 3,
        seed: 42,
        parallel: false,
        sim_config: SimulationConfig {
            dt: 1.0,
            max_time: 20.0,
            ..Default::default()
        },
        ..Default::default()
    };

    let state = State::new(
        Vector3::new(0.0, 0.0, 500.0),
        Vector3::new(0.0, 0.0, 0.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        vehicle.total_mass(),
        0.0,
    );

    let result = run_montecarlo_with_scenarios(
        &state, &vehicle, &aero, &controller, &scenarios, &config,
    );

    assert_eq!(result.n_total, 3);
    assert_eq!(result.individual_results.len(), 3);
}
