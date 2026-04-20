//! RK4 Integrator — Rigorous Verification (6-DOF State)

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rgnc::dynamics::eom::{State, StateDot};
use rgnc::dynamics::integrator::rk4_step;
use rgnc::aero::atmosphere;

const G_MARS: f64 = 3.7114;
const G_EARTH: f64 = 9.80665;

fn identity_state(pos: Vector3<f64>, vel: Vector3<f64>, mass: f64) -> State {
    State::new(pos, vel, UnitQuaternion::identity(), Vector3::zeros(), mass, 0.0)
}

// Translational-only derivative (no rotation) for polynomial tests
fn translational_deriv(gravity: f64) -> impl Fn(&State) -> StateDot {
    move |s: &State| StateDot {
        pos_dot: s.vel,
        vel_dot: Vector3::new(0.0, 0.0, gravity),
        quat_dot: Quaternion::new(0.0, 0.0, 0.0, 0.0),
        omega_dot: Vector3::zeros(),
        mass_dot: 0.0,
        time_dot: 1.0,
    }
}

#[test]
fn test_freefall_mars_rk4_exact() {
    let h0 = 2400.0;
    let dt = 0.1;
    let mut state = identity_state(Vector3::new(0.0, 0.0, h0), Vector3::zeros(), 1905.0);

    for _ in 0..100 {
        state = rk4_step(&state, dt, translational_deriv(-G_MARS));
        let t = state.time;
        let h_exact = h0 - 0.5 * G_MARS * t * t;
        let v_exact = -G_MARS * t;
        assert!((state.pos.z - h_exact).abs() < 1e-10,
            "t={:.1}: h err={:.2e}", t, (state.pos.z - h_exact).abs());
        assert!((state.vel.z - v_exact).abs() < 1e-10,
            "t={:.1}: v err={:.2e}", t, (state.vel.z - v_exact).abs());
    }
}

#[test]
fn test_freefall_earth_exact() {
    let h0 = 1000.0;
    let dt = 0.05;
    let mut state = identity_state(Vector3::new(0.0, 0.0, h0), Vector3::zeros(), 100.0);

    for _ in 0..200 {
        state = rk4_step(&state, dt, translational_deriv(-G_EARTH));
        let t = state.time;
        let h_exact = h0 - 0.5 * G_EARTH * t * t;
        assert!((state.pos.z - h_exact).abs() < 1e-10,
            "t={:.1}: h err={:.2e}", t, (state.pos.z - h_exact).abs());
    }
}

#[test]
fn test_3d_freefall_with_horizontal_velocity() {
    let h0 = 2400.0;
    let vx0 = 100.0;
    let vz0 = 50.0;
    let dt = 0.1;
    let mut state = identity_state(
        Vector3::new(0.0, 0.0, h0),
        Vector3::new(vx0, 0.0, vz0),
        1905.0,
    );

    for _ in 0..100 {
        state = rk4_step(&state, dt, translational_deriv(-G_MARS));
        let t = state.time;
        let x_exact = vx0 * t;
        let h_exact = h0 + vz0 * t - 0.5 * G_MARS * t * t;
        assert!((state.pos.x - x_exact).abs() < 1e-10);
        assert!((state.pos.z - h_exact).abs() < 1e-10);
    }
}

#[test]
fn test_rk4_fourth_order_convergence() {
    let y_exact = 0.5;
    let dts = [0.1, 0.05, 0.025, 0.0125];
    let mut errors = Vec::new();

    for &dt in &dts {
        let mut state = identity_state(Vector3::new(0.0, 0.0, 1.0), Vector3::zeros(), 1.0);
        let deriv = |s: &State| StateDot {
            pos_dot: Vector3::new(0.0, 0.0, -s.pos.z * s.pos.z),
            vel_dot: Vector3::zeros(),
            quat_dot: Quaternion::new(0.0, 0.0, 0.0, 0.0),
            omega_dot: Vector3::zeros(),
            mass_dot: 0.0,
            time_dot: 1.0,
        };
        let n = (1.0_f64 / dt) as usize;
        for _ in 0..n {
            state = rk4_step(&state, dt, &deriv);
        }
        errors.push((state.pos.z - y_exact).abs());
    }

    for i in 0..errors.len() - 1 {
        let ratio = errors[i] / errors[i + 1];
        assert!(ratio > 14.0 && ratio < 18.0,
            "Order check failed: ratio[{}] = {:.2}", i, ratio);
    }
}

#[test]
fn test_rk4_order_with_exponential_ode() {
    let y_exact = (-2.0_f64).exp();
    let dts = [0.1, 0.05, 0.025];
    let mut errors = Vec::new();

    for &dt in &dts {
        let mut state = identity_state(Vector3::new(0.0, 0.0, 1.0), Vector3::zeros(), 1.0);
        let deriv = |s: &State| StateDot {
            pos_dot: Vector3::new(0.0, 0.0, -s.pos.z),
            vel_dot: Vector3::zeros(),
            quat_dot: Quaternion::new(0.0, 0.0, 0.0, 0.0),
            omega_dot: Vector3::zeros(),
            mass_dot: 0.0,
            time_dot: 1.0,
        };
        let n = (2.0_f64 / dt) as usize;
        for _ in 0..n { state = rk4_step(&state, dt, &deriv); }
        errors.push((state.pos.z - y_exact).abs());
    }

    for i in 0..errors.len() - 1 {
        let ratio = errors[i] / errors[i + 1];
        assert!(ratio > 14.0 && ratio < 18.0,
            "Order check (exp ODE): ratio[{}] = {:.2}", i, ratio);
    }
}

#[test]
fn test_energy_conservation_projectile() {
    let mass = 22200.0;
    let mut state = identity_state(
        Vector3::new(0.0, 0.0, 50000.0),
        Vector3::new(433.0, 0.0, -250.0),
        mass,
    );

    let e0 = 0.5 * mass * state.vel.norm_squared() + mass * G_EARTH * state.pos.z;
    let dt = 0.01;

    for _ in 0..10000 {
        state = rk4_step(&state, dt, translational_deriv(-G_EARTH));
        let e = 0.5 * mass * state.vel.norm_squared() + mass * G_EARTH * state.pos.z;
        let rel_err = (e - e0).abs() / e0;
        assert!(rel_err < 1e-10,
            "energy drift at t={:.1}: rel_err = {:.2e}", state.time, rel_err);
    }
}

#[test]
fn test_mass_depletion_constant_thrust() {
    let m0 = 1905.0;
    let isp = 225.0;
    let thrust = 3100.0;
    let mdot = -thrust / (isp * G_EARTH);
    let dt = 0.1;

    let mut state = identity_state(
        Vector3::new(0.0, 0.0, 1500.0),
        Vector3::zeros(),
        m0,
    );

    let deriv = |s: &State| StateDot {
        pos_dot: s.vel,
        vel_dot: Vector3::new(0.0, 0.0, thrust / s.mass - G_MARS),
        quat_dot: Quaternion::new(0.0, 0.0, 0.0, 0.0),
        omega_dot: Vector3::zeros(),
        mass_dot: mdot,
        time_dot: 1.0,
    };

    for _ in 0..100 {
        state = rk4_step(&state, dt, &deriv);
        let t = state.time;
        let m_exact = m0 + mdot * t;
        assert!((state.mass - m_exact).abs() < 1e-8,
            "t={:.1}: mass err = {:.2e}", t, (state.mass - m_exact).abs());
        assert!(state.mass > 0.0);
    }
}

#[test]
fn test_quaternion_norm_preserved() {
    // Constant angular velocity rotation — quaternion norm must stay 1.0
    let omega = Vector3::new(0.1, 0.2, 0.3);
    let mut state = State::new(
        Vector3::zeros(), Vector3::zeros(),
        UnitQuaternion::identity(), omega,
        100.0, 0.0,
    );

    let deriv = |s: &State| {
        let q = s.quat.into_inner();
        let w = &s.omega;
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
            omega_dot: Vector3::zeros(), // no torque
            mass_dot: 0.0,
            time_dot: 1.0,
        }
    };

    for _ in 0..1000 { // 100 seconds at dt=0.1
        state = rk4_step(&state, 0.1, &deriv);
        let norm = state.quat.into_inner().norm();
        assert!((norm - 1.0).abs() < 1e-10,
            "quaternion norm drift at t={:.1}: {:.15}", state.time, norm);
    }
}

#[test]
fn test_drag_dissipates_energy() {
    let mass = 22200.0;
    let cd = 1.2;
    let s_ref = 10.0;

    let mut state = identity_state(
        Vector3::new(0.0, 0.0, 70000.0),
        Vector3::new(1477.0, 0.0, -260.4),
        mass,
    );

    let e0 = 0.5 * mass * state.vel.norm_squared() + mass * G_EARTH * state.pos.z;
    let mut prev_e = e0;
    let dt = 0.5;

    while state.pos.z > 5000.0 {
        let deriv = |s: &State| {
            let v = s.vel.norm();
            let atm = atmosphere(s.pos.z.max(0.0));
            let d = if v > 1e-10 {
                let q = 0.5 * atm.density * v * v;
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

        let e = 0.5 * mass * state.vel.norm_squared() + mass * G_EARTH * state.pos.z;
        assert!(e < prev_e + 1.0,
            "energy increased at t={:.1}: delta = {:.1} J", state.time, e - prev_e);
        prev_e = e;
    }

    let e_final = 0.5 * mass * state.vel.norm_squared() + mass * G_EARTH * state.pos.z;
    assert!(e_final < e0);
}
