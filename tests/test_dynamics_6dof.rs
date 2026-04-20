//! 6-DOF Dynamics Integration Tests

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rgnc::aero::atmosphere;
use rgnc::dynamics::eom::{State, StateDot};
use rgnc::dynamics::integrator::rk4_step;

const G_EARTH: f64 = 9.80665;
const G_MARS: f64 = 3.7114;

fn identity_state(pos: Vector3<f64>, vel: Vector3<f64>, mass: f64) -> State {
    State::new(pos, vel, UnitQuaternion::identity(), Vector3::zeros(), mass, 0.0)
}

#[test]
fn test_vertical_ascent_constant_thrust() {
    let mass = 1905.0;
    let thrust = 3100.0;
    let a_net = thrust / mass - G_MARS;
    let dt = 0.1;

    let mut state = identity_state(Vector3::zeros(), Vector3::zeros(), mass);

    for _ in 0..100 {
        let deriv = |s: &State| StateDot {
            pos_dot: s.vel,
            vel_dot: Vector3::new(0.0, 0.0, thrust / s.mass - G_MARS),
            quat_dot: Quaternion::new(0.0, 0.0, 0.0, 0.0),
            omega_dot: Vector3::zeros(),
            mass_dot: 0.0,
            time_dot: 1.0,
        };
        state = rk4_step(&state, dt, &deriv);
        let t = state.time;
        assert!((state.pos.z - 0.5 * a_net * t * t).abs() < 1e-8);
        assert!((state.vel.z - a_net * t).abs() < 1e-8);
    }
}

#[test]
fn test_ballistic_reentry_sanity() {
    let mass = 22200.0;
    let cd = 1.2;
    let s_ref = 10.0;
    let gamma = (-10.0_f64).to_radians();

    let mut state = identity_state(
        Vector3::new(0.0, 0.0, 70000.0),
        Vector3::new(1500.0 * gamma.cos(), 0.0, 1500.0 * gamma.sin()),
        mass,
    );

    let dt = 0.5;
    let mut peak_q = 0.0_f64;
    let mut peak_gload = 0.0_f64;

    while state.pos.z > 0.0 && state.time < 600.0 {
        let v = state.vel.norm();
        let atm = atmosphere(state.pos.z.max(0.0));
        let q = 0.5 * atm.density * v * v;
        peak_q = peak_q.max(q);

        let drag = q * cd * s_ref;
        let gload = (drag / mass + G_EARTH) / G_EARTH;
        peak_gload = peak_gload.max(gload);

        let deriv = |s: &State| {
            let v = s.vel.norm();
            let a = atmosphere(s.pos.z.max(0.0));
            let d = if v > 1e-10 {
                let q = 0.5 * a.density * v * v;
                -q * cd * s_ref / mass * s.vel / v
            } else { Vector3::zeros() };
            StateDot {
                pos_dot: s.vel,
                vel_dot: Vector3::new(0.0, 0.0, -G_EARTH) + d,
                quat_dot: Quaternion::new(0.0, 0.0, 0.0, 0.0),
                omega_dot: Vector3::zeros(),
                mass_dot: 0.0,
                time_dot: 1.0,
            }
        };
        state = rk4_step(&state, dt, &deriv);

        assert!(!state.pos.x.is_nan() && !state.pos.z.is_nan());
        assert!(!state.vel.x.is_nan() && !state.vel.z.is_nan());
    }

    assert!(state.time > 60.0 && state.time < 300.0,
        "flight time = {:.1} s", state.time);
    assert!(peak_q / 1000.0 > 30.0 && peak_q / 1000.0 < 300.0,
        "peak q = {:.1} kPa", peak_q / 1000.0);
    assert!(peak_gload > 2.0 && peak_gload < 10.0,
        "peak g-load = {:.1}", peak_gload);
}

#[test]
fn test_powered_descent_mass_conservation() {
    let m0 = 1905.0;
    let isp = 225.0;
    let thrust = 2000.0;
    let mdot = thrust / (isp * G_EARTH);
    let dt = 0.1;
    let t_final = 50.0;

    let mut state = identity_state(
        Vector3::new(0.0, 0.0, 1500.0),
        Vector3::new(-75.0, 0.0, 100.0),
        m0,
    );

    let n = (t_final / dt) as usize;
    for _ in 0..n {
        let deriv = |s: &State| StateDot {
            pos_dot: s.vel,
            vel_dot: Vector3::new(0.0, 0.0, thrust / s.mass - G_MARS),
            quat_dot: Quaternion::new(0.0, 0.0, 0.0, 0.0),
            omega_dot: Vector3::zeros(),
            mass_dot: -mdot,
            time_dot: 1.0,
        };
        state = rk4_step(&state, dt, &deriv);
    }

    let mass_consumed = m0 - state.mass;
    let expected_consumed = mdot * t_final;
    assert!((mass_consumed - expected_consumed).abs() < 1e-6);
    assert!(state.mass > 0.0);
}

#[test]
fn test_torque_free_symmetric_body() {
    // Symmetric body (Jx = Jy) with spin about z: omega should remain constant
    let omega = Vector3::new(0.0, 0.0, 1.0); // spin about body z
    let jx = 1000.0;
    let jy = 1000.0;
    let jz = 500.0;

    let mut state = State::new(
        Vector3::zeros(), Vector3::zeros(),
        UnitQuaternion::identity(), omega,
        100.0, 0.0,
    );

    let deriv = |s: &State| {
        let w = &s.omega;
        let jw = Vector3::new(jx * w.x, jy * w.y, jz * w.z);
        let omega_dot = Vector3::new(
            -(w.y * jw.z - w.z * jw.y) / jx,
            -(w.z * jw.x - w.x * jw.z) / jy,
            -(w.x * jw.y - w.y * jw.x) / jz,
        );
        let q = s.quat.into_inner();
        let quat_dot = Quaternion::new(
            -0.5 * (w.x * q.i + w.y * q.j + w.z * q.k),
             0.5 * (w.x * q.w + w.z * q.j - w.y * q.k),
             0.5 * (w.y * q.w - w.z * q.i + w.x * q.k),
             0.5 * (w.z * q.w + w.y * q.i - w.x * q.j),
        );
        StateDot {
            pos_dot: Vector3::zeros(),
            vel_dot: Vector3::zeros(),
            quat_dot,
            omega_dot,
            mass_dot: 0.0,
            time_dot: 1.0,
        }
    };

    for _ in 0..1000 {
        state = rk4_step(&state, 0.01, &deriv);
        // For symmetric body with pure z-spin: omega should remain constant
        assert!((state.omega.x).abs() < 1e-10,
            "omega_x drift at t={:.1}: {:.2e}", state.time, state.omega.x);
        assert!((state.omega.y).abs() < 1e-10,
            "omega_y drift at t={:.1}: {:.2e}", state.time, state.omega.y);
        assert!((state.omega.z - 1.0).abs() < 1e-10,
            "omega_z drift at t={:.1}: {:.2e}", state.time, (state.omega.z - 1.0).abs());
    }
}
