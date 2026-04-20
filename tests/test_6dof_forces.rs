//! Unit tests for 6-DOF force/moment computation functions.

use nalgebra::{UnitQuaternion, Vector3};
use rgnc::aero::{AeroTable, WindProfile};
use rgnc::dynamics::eom::*;
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

fn test_params<'a>(
    vehicle: &'a VehicleParams,
    aero: &'a AeroTable,
    wind: &'a WindProfile,
) -> DynamicsParams<'a> {
    DynamicsParams {
        vehicle,
        aero,
        wind,
        density_factor: 1.0,
        cd_factor: 1.0,
        thrust_bias: 0.0,
    }
}

fn default_state() -> State {
    State::new(
        Vector3::new(0.0, 0.0, 1000.0),
        Vector3::new(100.0, 0.0, 0.0),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0,
        0.0,
    )
}

// --- Gravity Tests ---

#[test]
fn test_gravity_no_moment() {
    let state = default_state();
    let (_, m) = compute_gravity(&state);
    assert!(m.norm() < 1e-15, "gravity should produce no moment");
}

#[test]
fn test_gravity_points_down() {
    let state = default_state();
    let (f, _) = compute_gravity(&state);
    assert!(f.z < 0.0, "gravity force should point down");
    assert!(f.x.abs() < 1e-15 && f.y.abs() < 1e-15, "no horizontal gravity");
}

#[test]
fn test_gravity_proportional_to_mass() {
    let state1 = State::new(
        Vector3::new(0.0, 0.0, 1000.0), Vector3::zeros(),
        UnitQuaternion::identity(), Vector3::zeros(), 100.0, 0.0,
    );
    let state2 = State::new(
        Vector3::new(0.0, 0.0, 1000.0), Vector3::zeros(),
        UnitQuaternion::identity(), Vector3::zeros(), 200.0, 0.0,
    );
    let (f1, _) = compute_gravity(&state1);
    let (f2, _) = compute_gravity(&state2);
    assert!((f2.z / f1.z - 2.0).abs() < 1e-10, "gravity should be proportional to mass");
}

// --- Thrust Tests ---

#[test]
fn test_thrust_zero_gimbal_no_moment() {
    // Engine on body -x axis, thrust along body +x: cross product = 0
    let state = default_state();
    let control = Control::new(1.0, 0.0, 0.0); // full throttle, no gimbal
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let wind = WindProfile::calm();
    let params = test_params(&vehicle, &aero, &wind);

    let (_, m, _) = compute_thrust(&state, &control, &params);
    // engine_offset = [-3, 0, 0], thrust_dir = [1, 0, 0]
    // moment = [-3, 0, 0] x [T, 0, 0] = [0, 0, 0]
    assert!(m.norm() < 1e-6,
        "zero gimbal should produce no thrust moment, got {:.6}", m.norm());
}

#[test]
fn test_thrust_gimbal_produces_moment() {
    let state = default_state();
    let control = Control::new(1.0, 0.1, 0.0); // pitch gimbal
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let wind = WindProfile::calm();
    let params = test_params(&vehicle, &aero, &wind);

    let (_, m, _) = compute_thrust(&state, &control, &params);
    assert!(m.norm() > 1.0,
        "gimbal should produce non-zero moment, got {:.6}", m.norm());
}

#[test]
fn test_thrust_throttle_linear() {
    let state = default_state();
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let wind = WindProfile::calm();
    let params = test_params(&vehicle, &aero, &wind);

    let ctrl_half = Control::new(0.5, 0.0, 0.0);
    let ctrl_full = Control::new(1.0, 0.0, 0.0);

    let (f_half, _, _) = compute_thrust(&state, &ctrl_half, &params);
    let (f_full, _, _) = compute_thrust(&state, &ctrl_full, &params);

    let t_half = vehicle.thrust_min + 0.5 * (vehicle.thrust_max - vehicle.thrust_min);
    let t_full = vehicle.thrust_max;
    let ratio = f_full.norm() / f_half.norm();
    let expected_ratio = t_full / t_half;
    assert!((ratio - expected_ratio).abs() < 0.01,
        "throttle should be linear: ratio={:.3}, expected={:.3}", ratio, expected_ratio);
}

#[test]
fn test_thrust_zero_throttle_no_force() {
    let state = default_state();
    let control = Control::zero();
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(0.0);
    let wind = WindProfile::calm();
    let params = test_params(&vehicle, &aero, &wind);

    let (f, m, mdot) = compute_thrust(&state, &control, &params);
    assert!(f.norm() < 1e-15, "zero throttle should produce no force");
    assert!(m.norm() < 1e-15, "zero throttle should produce no moment");
    assert!(mdot.abs() < 1e-15, "zero throttle should produce no mass flow");
}

// --- Aero Tests ---

#[test]
fn test_aero_zero_velocity_no_force() {
    let state = State::new(
        Vector3::new(0.0, 0.0, 1000.0),
        Vector3::zeros(), // zero velocity
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0, 0.0,
    );
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(1.2);
    let wind = WindProfile::calm();
    let params = test_params(&vehicle, &aero, &wind);

    let (f, m) = compute_aero(&state, &params);
    assert!(f.norm() < 1e-10, "zero velocity should produce no aero force");
    assert!(m.norm() < 1e-10, "zero velocity should produce no aero moment");
}

#[test]
fn test_aero_drag_opposes_motion() {
    let state = default_state(); // moving forward at 100 m/s
    let vehicle = test_vehicle();
    let aero = AeroTable::constant(1.2);
    let wind = WindProfile::calm();
    let params = test_params(&vehicle, &aero, &wind);

    let (f, _) = compute_aero(&state, &params);
    // Drag should oppose forward motion (negative x component)
    assert!(f.dot(&state.vel) < 0.0,
        "aero force should oppose velocity, dot={:.3}", f.dot(&state.vel));
}

#[test]
fn test_aero_moment_from_cp_offset() {
    // With CP ahead of CG (cp_offset.x > cg_offset.x) and forward flight at
    // a non-zero angle of attack, a non-zero lift force would produce a moment.
    // However, AeroTable::constant only sets cd (not cl), so lift = 0.
    // The drag is purely in body-x direction, r_cp = (2,0,0), so moment = 0.
    // To get a non-zero moment we need lateral force: use sideslip (y velocity).
    let state = State::new(
        Vector3::new(0.0, 0.0, 1000.0),
        Vector3::new(100.0, 10.0, 0.0), // sideslip velocity in y
        UnitQuaternion::identity(),
        Vector3::zeros(),
        1805.0, 0.0,
    );
    // Build a custom aero table with cy != 0 to produce a y-force
    let aero = AeroTable {
        mach_breaks: vec![0.0, 10.0],
        alpha_breaks: vec![0.0, 1.0],
        cd: vec![vec![1.2, 1.2], vec![1.2, 1.2]],
        cl: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        cy: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
    };
    let vehicle = test_vehicle();
    let wind = WindProfile::calm();
    let params = test_params(&vehicle, &aero, &wind);

    let (_, m) = compute_aero(&state, &params);
    // Sideslip creates Cy force in body-y direction; r_cp = (2,0,0)
    // moment = (2,0,0) x (fd_x, fd_y, 0) => non-zero z-component
    assert!(m.norm() > 0.0,
        "sideslip force with CP offset should produce non-zero aero moment, got {:.6}", m.norm());
}
