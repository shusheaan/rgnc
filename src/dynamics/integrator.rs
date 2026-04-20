use nalgebra::UnitQuaternion;
use super::eom::{State, StateDot};

/// RK4 integration step with quaternion normalization.
///
/// `deriv` computes the state derivative given the current state.
/// Quaternion is normalized after the final combination step.
pub fn rk4_step<F>(state: &State, dt: f64, deriv: F) -> State
where
    F: Fn(&State) -> StateDot,
{
    let k1 = deriv(state);
    let s2 = advance(state, &k1, dt * 0.5);
    let k2 = deriv(&s2);
    let s3 = advance(state, &k2, dt * 0.5);
    let k3 = deriv(&s3);
    let s4 = advance(state, &k3, dt);
    let k4 = deriv(&s4);

    let pos = state.pos + (dt / 6.0) * (k1.pos_dot + 2.0 * k2.pos_dot + 2.0 * k3.pos_dot + k4.pos_dot);
    let vel = state.vel + (dt / 6.0) * (k1.vel_dot + 2.0 * k2.vel_dot + 2.0 * k3.vel_dot + k4.vel_dot);
    let omega = state.omega + (dt / 6.0) * (k1.omega_dot + 2.0 * k2.omega_dot + 2.0 * k3.omega_dot + k4.omega_dot);
    let mass = state.mass + (dt / 6.0) * (k1.mass_dot + 2.0 * k2.mass_dot + 2.0 * k3.mass_dot + k4.mass_dot);

    let quat_raw = state.quat.into_inner()
        + (dt / 6.0) * (k1.quat_dot + 2.0 * k2.quat_dot + 2.0 * k3.quat_dot + k4.quat_dot);
    let quat = UnitQuaternion::new_normalize(quat_raw);

    State {
        pos,
        vel,
        quat,
        omega,
        mass,
        time: state.time + dt,
    }
}

fn advance(state: &State, dot: &StateDot, dt: f64) -> State {
    let quat_raw = state.quat.into_inner() + dt * dot.quat_dot;
    State {
        pos: state.pos + dt * dot.pos_dot,
        vel: state.vel + dt * dot.vel_dot,
        quat: UnitQuaternion::new_normalize(quat_raw),
        omega: state.omega + dt * dot.omega_dot,
        mass: state.mass + dt * dot.mass_dot,
        time: state.time + dt * dot.time_dot,
    }
}
