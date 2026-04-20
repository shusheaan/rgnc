/// Integration tests for the closed-loop guidance pipeline.
///
/// Tests the full pipeline: reference trajectory generation,
/// trajectory library, closed-loop controller, and comparison
/// between open-loop and closed-loop performance.

use nalgebra::{UnitQuaternion, Vector3};
use rgnc::aero::AeroTable;
use rgnc::dynamics::State;
use rgnc::mission::closed_loop::*;
use rgnc::mission::simulate::*;
use rgnc::robust::Scenario;
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

fn landing_state() -> State {
    State::new(
        Vector3::new(0.0, 0.0, 2400.0),
        Vector3::new(0.0, 0.0, -75.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0,
        0.0,
    )
}

#[test]
fn test_generate_reference_trajectory() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let state = landing_state();

    let ref_traj = generate_reference_trajectory(&state, &vehicle, &aero, 0.1);

    // Reference trajectory should have many points
    assert!(
        ref_traj.states.len() > 10,
        "Reference trajectory should have >10 points, got {}",
        ref_traj.states.len()
    );

    // Should start at the initial state
    let first = &ref_traj.states[0];
    assert!((first.pos - state.pos).norm() < 0.1);

    // Should end near ground
    let last = ref_traj.states.last().unwrap();
    assert!(
        last.pos.z < 100.0,
        "Reference should reach near ground: final_alt={:.1}m",
        last.pos.z
    );
}

#[test]
fn test_trajectory_library_selection() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);

    // Create library with two trajectories: one from (0,0,2400) and one from (500,0,2400)
    let state1 = State::new(
        Vector3::new(0.0, 0.0, 2400.0),
        Vector3::new(0.0, 0.0, -75.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0, 0.0,
    );
    let state2 = State::new(
        Vector3::new(500.0, 0.0, 2400.0),
        Vector3::new(0.0, 0.0, -75.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0, 0.0,
    );

    let traj1 = generate_reference_trajectory(&state1, &vehicle, &aero, 0.5);
    let mut traj2 = generate_reference_trajectory(&state2, &vehicle, &aero, 0.5);
    traj2.id = 1;

    let mut library = TrajectoryLibrary::new();
    library.add(traj1);
    library.add(traj2);

    // Query from near (0,0,2400) - should select trajectory 0
    let query = State::new(
        Vector3::new(10.0, 0.0, 2400.0),
        Vector3::new(0.0, 0.0, -75.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0, 0.0,
    );
    let selected = library.select(&query).unwrap();
    assert_eq!(selected.id, 0, "Should select closest trajectory");

    // Query from near (500,0,2400) - should select trajectory 1
    let query2 = State::new(
        Vector3::new(490.0, 0.0, 2400.0),
        Vector3::new(0.0, 0.0, -75.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0, 0.0,
    );
    let selected2 = library.select(&query2).unwrap();
    assert_eq!(selected2.id, 1, "Should select closest trajectory");
}

#[test]
fn test_closed_loop_controller_runs() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let state = landing_state();

    // Build a simple library
    let ref_traj = generate_reference_trajectory(&state, &vehicle, &aero, 0.1);
    let mut library = TrajectoryLibrary::new();
    library.add(ref_traj);

    let scenario = Scenario::nominal(0);
    let config = SimulationConfig {
        dt: 0.1,
        max_time: 200.0,
        ..Default::default()
    };

    let result = closed_loop_simulate(&state, &vehicle, &aero, &scenario, &library, &config);

    assert!(result.landed, "Closed-loop should reach ground");
    assert!(result.fuel_used > 0.0, "Should use fuel");
}

#[test]
fn test_closed_loop_handles_wind_perturbation() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let state = landing_state();

    // Build library under nominal conditions
    let ref_traj = generate_reference_trajectory(&state, &vehicle, &aero, 0.1);
    let mut library = TrajectoryLibrary::new();
    library.add(ref_traj);

    let config = SimulationConfig {
        dt: 0.1,
        max_time: 200.0,
        ..Default::default()
    };

    // Test under wind perturbation
    let scenario_windy = Scenario {
        id: 1,
        wind_profile: vec![
            (0.0, Vector3::new(5.0, 0.0, 0.0)),
            (100_000.0, Vector3::new(5.0, 0.0, 0.0)),
        ],
        density_factor: 1.0,
        cd_factor: 1.0,
        thrust_bias: 0.0,
    };

    let result = closed_loop_simulate(&state, &vehicle, &aero, &scenario_windy, &library, &config);
    assert!(result.landed, "Should still land under moderate wind");
}

#[test]
fn test_full_pipeline_nominal_vs_perturbed() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let state = landing_state();

    // Step 1: Generate reference under nominal
    let ref_traj = generate_reference_trajectory(&state, &vehicle, &aero, 0.1);
    let mut library = TrajectoryLibrary::new();
    library.add(ref_traj);

    let config = SimulationConfig {
        dt: 0.1,
        max_time: 200.0,
        ..Default::default()
    };

    // Step 2: Run under nominal
    let nominal_result = closed_loop_simulate(
        &state, &vehicle, &aero, &Scenario::nominal(0), &library, &config,
    );

    // Step 3: Run under perturbed conditions
    let perturbed = Scenario {
        id: 1,
        wind_profile: vec![
            (0.0, Vector3::new(3.0, 2.0, 0.0)),
            (100_000.0, Vector3::new(3.0, 2.0, 0.0)),
        ],
        density_factor: 1.05,
        cd_factor: 1.1,
        thrust_bias: 20.0,
    };
    let perturbed_result = closed_loop_simulate(
        &state, &vehicle, &aero, &perturbed, &library, &config,
    );

    // Both should land
    assert!(nominal_result.landed, "Nominal should land");
    assert!(perturbed_result.landed, "Perturbed should land");

    // Perturbed might use different fuel
    assert!(
        (perturbed_result.fuel_used - nominal_result.fuel_used).abs() > 0.01
            || (perturbed_result.landing_error - nominal_result.landing_error).abs() > 0.01,
        "Perturbations should affect results"
    );
}

#[test]
fn test_reference_trajectory_control_lookup() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let state = landing_state();

    let ref_traj = generate_reference_trajectory(&state, &vehicle, &aero, 0.5);

    // Control at time 0 should be valid
    let _ctrl0 = ref_traj.control_at_time(0.0);
    // Shouldn't panic

    // Control beyond trajectory end should clamp
    let _ctrl_late = ref_traj.control_at_time(10000.0);
    // Shouldn't panic

    // State lookup
    let state0 = ref_traj.state_at_time(0.0);
    assert!((state0.pos - state.pos).norm() < 0.1);
}

#[test]
fn test_empty_library_falls_back() {
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let state = landing_state();

    // Empty library: should use fallback controller
    let library = TrajectoryLibrary::new();
    let scenario = Scenario::nominal(0);
    let config = SimulationConfig {
        dt: 0.1,
        max_time: 200.0,
        ..Default::default()
    };

    let result = closed_loop_simulate(&state, &vehicle, &aero, &scenario, &library, &config);

    // Should still run without crashing (uses fallback)
    assert!(result.trajectory.len() > 1, "Should produce a trajectory");
}
