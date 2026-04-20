use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use crate::aero::{atmosphere, AeroTable, WindProfile};
use crate::vehicle::VehicleParams;

/// Standard gravitational acceleration (m/s²).
pub const G0_EARTH: f64 = 9.80665;
/// Mean Earth radius (m).
pub const R_EARTH: f64 = 6_371_000.0;

/// Full state vector for 6-DOF rigid-body dynamics.
///
/// Coordinate frame: flat-Earth NED-up (x=downrange, y=crossrange, z=up).
/// Body frame: x=forward (nose), y=right, z=down.
#[derive(Debug, Clone)]
pub struct State {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub quat: UnitQuaternion<f64>,
    pub omega: Vector3<f64>,
    pub mass: f64,
    pub time: f64,
}

/// Time derivative of the 6-DOF state vector.
#[derive(Debug, Clone)]
pub struct StateDot {
    pub pos_dot: Vector3<f64>,
    pub vel_dot: Vector3<f64>,
    pub quat_dot: Quaternion<f64>,
    pub omega_dot: Vector3<f64>,
    pub mass_dot: f64,
    pub time_dot: f64,
}

/// Control input: throttle + gimbal angles.
#[derive(Debug, Clone)]
pub struct Control {
    /// Throttle [0, 1]: actual thrust = thrust_min + throttle * (thrust_max - thrust_min)
    pub throttle: f64,
    /// Pitch gimbal deflection (rad), clamped to [-gimbal_max, gimbal_max]
    pub gimbal_y: f64,
    /// Yaw gimbal deflection (rad), clamped to [-gimbal_max, gimbal_max]
    pub gimbal_z: f64,
}

impl State {
    pub fn new(
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        quat: UnitQuaternion<f64>,
        omega: Vector3<f64>,
        mass: f64,
        time: f64,
    ) -> Self {
        Self { pos, vel, quat, omega, mass, time }
    }

    /// Altitude (z-component of position, NED-up convention).
    pub fn altitude(&self) -> f64 {
        self.pos.z
    }

    /// Speed (magnitude of velocity).
    pub fn speed(&self) -> f64 {
        self.vel.norm()
    }

    /// Dynamic pressure (Pa).
    pub fn dynamic_pressure(&self) -> f64 {
        let atm = atmosphere(self.altitude());
        0.5 * atm.density * self.speed().powi(2)
    }
}

impl Control {
    pub fn zero() -> Self {
        Self { throttle: 0.0, gimbal_y: 0.0, gimbal_z: 0.0 }
    }

    pub fn new(throttle: f64, gimbal_y: f64, gimbal_z: f64) -> Self {
        Self { throttle, gimbal_y, gimbal_z }
    }
}

/// Parameters for the 6-DOF dynamics computation.
pub struct DynamicsParams<'a> {
    pub vehicle: &'a VehicleParams,
    pub aero: &'a AeroTable,
    pub wind: &'a WindProfile,
    pub density_factor: f64,
    pub cd_factor: f64,
    pub thrust_bias: f64,
}

/// Compute gravitational force in inertial frame.
/// Returns (force_inertial, moment_body). Gravity has no moment about CG.
pub fn compute_gravity(state: &State) -> (Vector3<f64>, Vector3<f64>) {
    let h = state.pos.z.max(0.0);
    let g_factor = (R_EARTH / (R_EARTH + h)).powi(2);
    let f_grav = state.mass * Vector3::new(0.0, 0.0, -G0_EARTH * g_factor);
    (f_grav, Vector3::zeros())
}

/// Compute aerodynamic force (inertial) and moment (body).
///
/// Angle of attack and sideslip are computed from attitude and relative velocity.
/// Force is rotated to inertial frame; moment arises from CP-CG offset.
pub fn compute_aero(state: &State, params: &DynamicsParams) -> (Vector3<f64>, Vector3<f64>) {
    let h = state.pos.z.max(0.0);
    let atm = atmosphere(h);
    let rho = atm.density * params.density_factor;

    let wind = params.wind.at_altitude(h);
    let v_rel = state.vel - wind;
    let v_rel_mag = v_rel.norm();

    if v_rel_mag < 1e-6 {
        return (Vector3::zeros(), Vector3::zeros());
    }

    // Transform relative velocity to body frame
    let v_body = state.quat.inverse_transform_vector(&v_rel);

    // Angle of attack and sideslip
    let alpha = v_body.z.atan2(v_body.x);
    let beta = (v_body.y / v_rel_mag).asin();

    let mach = v_rel_mag / atm.speed_of_sound;
    let (cd, cl) = params.aero.lookup(mach, alpha.abs());
    let cd = cd * params.cd_factor;
    let cy = params.aero.lookup_cy(mach, beta.abs());

    let q_dyn = 0.5 * rho * v_rel_mag * v_rel_mag;
    let s = params.vehicle.ref_area;

    // Aerodynamic force in body frame
    // Drag opposes body-x (forward), lift in body-z plane, sideforce in body-y
    let f_aero_body = q_dyn * s * Vector3::new(
        -cd,
        cy * beta.signum(),
        -cl * alpha.signum(),
    );

    // Rotate to inertial frame
    let f_aero_inertial = state.quat.transform_vector(&f_aero_body);

    // Aerodynamic moment from CP-CG offset (body frame)
    let r_cp = params.vehicle.cp_offset - params.vehicle.cg_offset;
    let m_aero = r_cp.cross(&f_aero_body);

    (f_aero_inertial, m_aero)
}

/// Compute thrust force (inertial) and moment (body).
///
/// Thrust direction determined by gimbal angles, applied at engine mount point.
pub fn compute_thrust(
    state: &State,
    control: &Control,
    params: &DynamicsParams,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    if control.throttle <= 0.0 {
        return (Vector3::zeros(), Vector3::zeros(), 0.0);
    }

    let gmax = params.vehicle.gimbal_max;
    let gy = control.gimbal_y.clamp(-gmax, gmax);
    let gz = control.gimbal_z.clamp(-gmax, gmax);

    // Thrust direction in body frame
    let e_thrust = Vector3::new(
        gy.cos() * gz.cos(),
        -(gy.cos() * gz.sin()),
        -gy.sin(),
    ).normalize();

    // Thrust magnitude
    let t_mag = params.vehicle.thrust_min
        + control.throttle * (params.vehicle.thrust_max - params.vehicle.thrust_min)
        + params.thrust_bias;
    let t_mag = t_mag.max(0.0);

    let f_thrust_body = t_mag * e_thrust;
    let f_thrust_inertial = state.quat.transform_vector(&f_thrust_body);

    // Thrust moment from engine offset (body frame)
    let m_thrust = params.vehicle.engine_offset.cross(&f_thrust_body);

    // Mass flow rate
    let mass_dot = -t_mag / (params.vehicle.isp * G0_EARTH);

    (f_thrust_inertial, m_thrust, mass_dot)
}

/// Compute 6-DOF rigid-body derivatives.
///
/// Uses force/moment accumulator pattern:
/// - Gravity: inertial force, no moment
/// - Aerodynamics: force (inertial) + moment (body) from CP-CG offset
/// - Thrust: force (inertial) + moment (body) from engine offset + gimbal
///
/// Rotation: Euler equations J*omega_dot = M_total - omega x (J*omega)
/// Attitude: quaternion kinematics q_dot = 0.5 * q * [0, omega]
pub fn derivatives_6dof(state: &State, control: &Control, params: &DynamicsParams) -> StateDot {
    let (f_grav, _) = compute_gravity(state);
    let (f_aero, m_aero) = compute_aero(state, params);
    let (f_thrust, m_thrust, mass_dot) = compute_thrust(state, control, params);

    // Translational dynamics: F = m*a
    let f_total = f_grav + f_aero + f_thrust;
    let vel_dot = f_total / state.mass;

    // Rotational dynamics: Euler equations
    let j = &params.vehicle.inertia;
    let omega = &state.omega;
    let j_omega = Vector3::new(j.x * omega.x, j.y * omega.y, j.z * omega.z);
    let m_total = m_aero + m_thrust;
    let omega_dot = Vector3::new(
        (m_total.x - (omega.y * j_omega.z - omega.z * j_omega.y)) / j.x,
        (m_total.y - (omega.z * j_omega.x - omega.x * j_omega.z)) / j.y,
        (m_total.z - (omega.x * j_omega.y - omega.y * j_omega.x)) / j.z,
    );

    // Quaternion kinematics: q_dot = 0.5 * q * [0, omega]
    let q = state.quat.into_inner();
    let quat_dot = Quaternion::new(
        -0.5 * (omega.x * q.i + omega.y * q.j + omega.z * q.k),
         0.5 * (omega.x * q.w + omega.z * q.j - omega.y * q.k),
         0.5 * (omega.y * q.w - omega.z * q.i + omega.x * q.k),
         0.5 * (omega.z * q.w + omega.y * q.i - omega.x * q.j),
    );

    StateDot {
        pos_dot: state.vel,
        vel_dot,
        quat_dot,
        omega_dot,
        mass_dot,
        time_dot: 1.0,
    }
}
