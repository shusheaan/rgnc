//! Successive Convexification (SCvx) algorithm for 6-DOF PDG.
//!
//! Ref: Szmuk & Açıkmeşe (2018), Malyuta et al. (2022).
//!
//! # Algorithm
//!
//! ```text
//! 1. Initialize reference trajectory x̄ (linear interp), controls ū (hover)
//! 2. For each iteration:
//!    a. Linearize: compute A_k, B_k, f_k at (x̄, ū)
//!    b. Solve convex SOCP subproblem → (x*, u*)
//!    c. Forward-simulate with nonlinear dynamics using u*
//!    d. Compute defect = ‖x_sim - x*‖ and virtual control norm
//!    e. Trust region update:
//!       - defect small → expand (×2), defect large → shrink (×0.5)
//!    f. Convergence: vc_norm < tol AND defect_pos < 1m AND defect_vel < 0.5m/s
//! 3. Return converged trajectory or max-iteration result
//! ```

use nalgebra::{DVector, UnitQuaternion, Quaternion, Vector3};
use crate::dynamics::eom::State;
use crate::guidance::linearize::*;
use crate::guidance::scvx_formulation::build_scvx_subproblem;
use crate::solver::socp::solve_socp;
use crate::solver::result::SolveStatusGeneric;

#[derive(Debug, Clone)]
pub struct ScvxConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub trust_region_radius: f64,
    pub trust_region_shrink: f64,
    pub trust_region_expand: f64,
}

impl Default for ScvxConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            trust_region_radius: 1.0,
            trust_region_shrink: 0.5,
            trust_region_expand: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScvxProblem {
    pub vehicle: ScvxVehicleParams,
    pub drag: DragParams,
    pub gravity: f64,
    pub initial_pos: Vector3<f64>,
    pub initial_vel: Vector3<f64>,
    pub wet_mass: f64,
    pub dry_mass: f64,
    pub n_timesteps: usize,
    pub dt: f64,
    pub config: ScvxConfig,
}

#[derive(Debug, Clone)]
pub struct ScvxIteration {
    pub cost: f64,
    pub defect_pos: f64,
    pub defect_vel: f64,
    pub vc_norm: f64,
    pub trust_radius: f64,
    pub accepted: bool,
}

#[derive(Debug, Clone)]
pub struct ScvxSolution {
    pub trajectory: Vec<State>,
    pub fuel_used: f64,
    pub converged: bool,
    pub iterations: usize,
    pub history: Vec<ScvxIteration>,
}

const NX: usize = 14;
const NU: usize = 4;
const NVC: usize = 10;

#[allow(dead_code)]
fn state_to_dvec(s: &State) -> DVector<f64> {
    let q = s.quat.into_inner();
    DVector::from_vec(vec![
        s.pos.x, s.pos.y, s.pos.z,
        s.vel.x, s.vel.y, s.vel.z,
        q.w, q.i, q.j, q.k,
        s.omega.x, s.omega.y, s.omega.z,
        s.mass.ln(),
    ])
}

fn dvec_to_state(x: &DVector<f64>, time: f64) -> State {
    State::new(
        Vector3::new(x[0], x[1], x[2]),
        Vector3::new(x[3], x[4], x[5]),
        UnitQuaternion::new_normalize(Quaternion::new(x[6], x[7], x[8], x[9])),
        Vector3::new(x[10], x[11], x[12]),
        x[13].exp(),
        time,
    )
}

pub fn solve_scvx(prob: &ScvxProblem) -> ScvxSolution {
    let n = prob.n_timesteps;
    let dt = prob.dt;

    // Initialize reference trajectory (linear interpolation)
    let mut x_ref = Vec::with_capacity(n + 1);
    let mut u_ref = Vec::with_capacity(n);
    let z0 = prob.wet_mass.ln();

    for k in 0..=n {
        let frac = k as f64 / n as f64;
        let mut xk = DVector::zeros(NX);
        let r = prob.initial_pos * (1.0 - frac);
        let v = prob.initial_vel * (1.0 - frac);
        xk[0] = r.x; xk[1] = r.y; xk[2] = r.z;
        xk[3] = v.x; xk[4] = v.y; xk[5] = v.z;
        xk[6] = 1.0; // identity quaternion
        xk[13] = z0;
        x_ref.push(xk);
    }
    for _ in 0..n {
        let mut uk = DVector::zeros(NU);
        uk[2] = prob.gravity; // hover thrust in z
        uk[3] = prob.gravity; // sigma = |T|
        u_ref.push(uk);
    }

    let mut trust_radius = prob.config.trust_region_radius;
    let mut history = Vec::new();
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..prob.config.max_iterations {
        iterations = iter + 1;

        // 1. Linearize around reference
        let lin = linearize_trajectory(&x_ref, &u_ref, prob.gravity, &prob.drag, &prob.vehicle);

        // 2. Build and solve SOCP subproblem
        let mut config_iter = prob.config.clone();
        config_iter.trust_region_radius = trust_radius;
        let data = build_scvx_subproblem(
            &x_ref, &u_ref, &lin, dt, prob.gravity, &prob.drag, &prob.vehicle, &config_iter,
        );
        let result = solve_socp(&data, false);

        if result.status != SolveStatusGeneric::Solved {
            trust_radius *= prob.config.trust_region_shrink;
            history.push(ScvxIteration {
                cost: f64::INFINITY, defect_pos: f64::INFINITY,
                defect_vel: f64::INFINITY, vc_norm: f64::INFINITY,
                trust_radius, accepted: false,
            });
            continue;
        }

        // 3. Extract solution
        let sol = &result.primal;
        let ctrl_off = NX * (n + 1);
        let vc_off = ctrl_off + NU * n;

        let mut x_opt = Vec::with_capacity(n + 1);
        let mut u_opt = Vec::with_capacity(n);
        for k in 0..=n {
            x_opt.push(DVector::from_column_slice(&sol[NX * k..NX * (k + 1)]));
        }
        for k in 0..n {
            u_opt.push(DVector::from_column_slice(&sol[ctrl_off + NU * k..ctrl_off + NU * (k + 1)]));
        }

        // 4. Virtual control norm
        let mut vc_norm = 0.0;
        for k in 0..n {
            for j in 0..NVC {
                vc_norm += sol[vc_off + NVC * k + j].abs();
            }
        }

        // 5. Forward simulate with nonlinear dynamics (Euler)
        let mut x_sim = vec![x_opt[0].clone()];
        for k in 0..n {
            let f_k = scvx_dynamics_full(&x_sim[k], &u_opt[k], prob.gravity, &prob.drag, &prob.vehicle);
            let x_next = &x_sim[k] + dt * f_k;
            x_sim.push(x_next);
        }

        // 6. Compute defect
        let mut defect_pos = 0.0f64;
        let mut defect_vel = 0.0f64;
        for k in 0..=n {
            for i in 0..3 {
                defect_pos = defect_pos.max((x_sim[k][i] - x_opt[k][i]).abs());
            }
            for i in 3..6 {
                defect_vel = defect_vel.max((x_sim[k][i] - x_opt[k][i]).abs());
            }
        }

        let cost = -x_opt[n][13];

        history.push(ScvxIteration {
            cost, defect_pos, defect_vel, vc_norm, trust_radius, accepted: true,
        });

        // 7. Update reference
        x_ref = x_opt;
        u_ref = u_opt;

        // 8. Adjust trust region
        if defect_pos < 10.0 && defect_vel < 5.0 {
            trust_radius = (trust_radius * prob.config.trust_region_expand).min(1000.0);
        } else if defect_pos > 100.0 || defect_vel > 50.0 {
            trust_radius *= prob.config.trust_region_shrink;
        }

        // 9. Convergence check
        if vc_norm < prob.config.tolerance && defect_pos < 1.0 && defect_vel < 0.5 {
            converged = true;
            break;
        }
    }

    let trajectory: Vec<State> = x_ref.iter().enumerate()
        .map(|(k, xk)| dvec_to_state(xk, k as f64 * dt))
        .collect();
    let fuel_used = prob.wet_mass - trajectory.last().unwrap().mass;

    ScvxSolution { trajectory, fuel_used, converged, iterations, history }
}
