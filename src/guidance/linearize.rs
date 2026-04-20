//! Dynamics linearization for 6-DOF SCvx.
//!
//! Ref: Szmuk & Açıkmeşe (2018) "SCvx for 6-DoF Mars Rocket Powered Landing"
//!
//! # SCvx Dynamics (14D state, 4D control)
//!
//! ```text
//! State x = [r(3), v(3), q(4), ω(3), z(1)]  where z = ln(m)
//! Control u = [T(3), σ(1)]  (inertial thrust accel + magnitude slack)
//!
//! ṙ = v
//! v̇ = T + g + a_drag       where a_drag = -½ρ·Cd·S·‖v‖·v̂/m
//! q̇ = ½·q⊗[0,ω]           (quaternion kinematics, Hamilton)
//! ω̇ = J⁻¹(τ - ω×Jω)      (Euler equations)
//! ż = -α·σ                  where α = 1/(Isp·g₀)
//! ```
//!
//! Jacobians A_k = ∂f/∂x, B_k = ∂f/∂u computed via central finite differences:
//!   ∂f/∂x_j ≈ (f(x+εe_j) - f(x-εe_j)) / 2ε,  ε = 1e-7

use nalgebra::{DMatrix, DVector, UnitQuaternion, Quaternion, Vector3};

/// Aerodynamic drag parameters for SCvx.
#[derive(Debug, Clone)]
pub struct DragParams {
    pub ref_area: f64,
    pub cd: f64,
    pub density: f64,
}

/// Vehicle parameters for SCvx rotational dynamics.
#[derive(Debug, Clone)]
pub struct ScvxVehicleParams {
    pub inertia: Vector3<f64>,
    pub isp: f64,
    pub thrust_max: f64,
    pub cp_offset: Vector3<f64>,
}

const G0_EARTH: f64 = 9.80665;

/// SCvx state layout (14D):
///   [0..3]   r  — position (inertial)
///   [3..6]   v  — velocity (inertial)
///   [6..10]  q  — quaternion [w, x, y, z]
///   [10..13] w  — angular velocity (body frame)
///   [13]     z  — log-mass ln(m)
///
/// SCvx control layout (4D):
///   [0..3]  T     — thrust acceleration (inertial frame, m/s^2)
///   [3]     sigma — thrust magnitude slack
pub fn scvx_dynamics(
    x: &DVector<f64>,
    u: &DVector<f64>,
    gravity: f64,
    drag: &DragParams,
) -> DVector<f64> {
    scvx_dynamics_full(x, u, gravity, drag, &default_vehicle_params())
}

pub fn scvx_dynamics_full(
    x: &DVector<f64>,
    u: &DVector<f64>,
    gravity: f64,
    drag: &DragParams,
    veh: &ScvxVehicleParams,
) -> DVector<f64> {
    let v = Vector3::new(x[3], x[4], x[5]);
    let q = UnitQuaternion::new_normalize(Quaternion::new(x[6], x[7], x[8], x[9]));
    let w = Vector3::new(x[10], x[11], x[12]);
    let z = x[13];
    let mass = z.exp();

    let t_accel = Vector3::new(u[0], u[1], u[2]);
    let sigma = u[3];
    let alpha = 1.0 / (veh.isp * G0_EARTH);

    // Position: r_dot = v
    let r_dot = v;

    // Velocity: v_dot = thrust + gravity + drag
    let g_vec = Vector3::new(0.0, 0.0, -gravity);
    let v_mag = v.norm();
    let a_drag = if v_mag > 1e-6 && drag.density > 0.0 {
        let q_dyn = 0.5 * drag.density * v_mag * v_mag;
        let drag_accel = q_dyn * drag.cd * drag.ref_area / mass;
        -drag_accel * v / v_mag
    } else {
        Vector3::zeros()
    };
    let v_dot = t_accel + g_vec + a_drag;

    // Quaternion: q_dot = 0.5 * q * [0, w]
    let qv = q.into_inner();
    let q_dot = Quaternion::new(
        -0.5 * (w.x * qv.i + w.y * qv.j + w.z * qv.k),
         0.5 * (w.x * qv.w + w.z * qv.j - w.y * qv.k),
         0.5 * (w.y * qv.w - w.z * qv.i + w.x * qv.k),
         0.5 * (w.z * qv.w + w.y * qv.i - w.x * qv.j),
    );

    // Angular velocity: Euler equations (no external moments for first version)
    let j = &veh.inertia;
    let jw = Vector3::new(j.x * w.x, j.y * w.y, j.z * w.z);
    let m_total = Vector3::<f64>::zeros();
    let w_dot = Vector3::new(
        (m_total.x - (w.y * jw.z - w.z * jw.y)) / j.x,
        (m_total.y - (w.z * jw.x - w.x * jw.z)) / j.y,
        (m_total.z - (w.x * jw.y - w.y * jw.x)) / j.z,
    );

    // Log-mass: z_dot = -alpha * sigma
    let z_dot = -alpha * sigma;

    let mut xdot = DVector::zeros(14);
    xdot[0] = r_dot.x; xdot[1] = r_dot.y; xdot[2] = r_dot.z;
    xdot[3] = v_dot.x; xdot[4] = v_dot.y; xdot[5] = v_dot.z;
    xdot[6] = q_dot.w;  xdot[7] = q_dot.i;  xdot[8] = q_dot.j;  xdot[9] = q_dot.k;
    xdot[10] = w_dot.x; xdot[11] = w_dot.y; xdot[12] = w_dot.z;
    xdot[13] = z_dot;
    xdot
}

/// Compute Jacobians df/dx (14x14) and df/du (14x4) via central finite differences.
pub fn compute_jacobians(
    x: &DVector<f64>,
    u: &DVector<f64>,
    gravity: f64,
    drag: &DragParams,
) -> (DMatrix<f64>, DMatrix<f64>) {
    compute_jacobians_full(x, u, gravity, drag, &default_vehicle_params())
}

/// Compute Jacobians with full vehicle parameters.
pub fn compute_jacobians_full(
    x: &DVector<f64>,
    u: &DVector<f64>,
    gravity: f64,
    drag: &DragParams,
    veh: &ScvxVehicleParams,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let nx = 14;
    let nu = 4;
    let eps = 1e-7;

    let mut a = DMatrix::zeros(nx, nx);
    for j in 0..nx {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += eps;
        xm[j] -= eps;
        let fp = scvx_dynamics_full(&xp, u, gravity, drag, veh);
        let fm = scvx_dynamics_full(&xm, u, gravity, drag, veh);
        let col = (fp - fm) / (2.0 * eps);
        a.set_column(j, &col);
    }

    let mut b = DMatrix::zeros(nx, nu);
    for j in 0..nu {
        let mut up = u.clone();
        let mut um = u.clone();
        up[j] += eps;
        um[j] -= eps;
        let fp = scvx_dynamics_full(x, &up, gravity, drag, veh);
        let fm = scvx_dynamics_full(x, &um, gravity, drag, veh);
        let col = (fp - fm) / (2.0 * eps);
        b.set_column(j, &col);
    }

    (a, b)
}

/// Linearization data for one SCvx iteration (all timesteps).
pub struct Linearization {
    pub a: Vec<DMatrix<f64>>,
    pub b: Vec<DMatrix<f64>>,
    pub f_ref: Vec<DVector<f64>>,
}

/// Compute linearization at all timesteps of a reference trajectory.
pub fn linearize_trajectory(
    x_ref: &[DVector<f64>],
    u_ref: &[DVector<f64>],
    gravity: f64,
    drag: &DragParams,
    veh: &ScvxVehicleParams,
) -> Linearization {
    let n = u_ref.len();
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut f_ref = Vec::with_capacity(n);

    for k in 0..n {
        let (ak, bk) = compute_jacobians_full(&x_ref[k], &u_ref[k], gravity, drag, veh);
        let fk = scvx_dynamics_full(&x_ref[k], &u_ref[k], gravity, drag, veh);
        a.push(ak);
        b.push(bk);
        f_ref.push(fk);
    }

    Linearization { a, b, f_ref }
}

fn default_vehicle_params() -> ScvxVehicleParams {
    ScvxVehicleParams {
        inertia: Vector3::new(1000.0, 1000.0, 200.0),
        isp: 225.0,
        thrust_max: 18600.0,
        cp_offset: Vector3::new(2.0, 0.0, 0.0),
    }
}
