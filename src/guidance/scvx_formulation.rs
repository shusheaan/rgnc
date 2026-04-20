//! SOCP subproblem construction for 6-DOF SCvx.
//!
//! Ref: Szmuk & Açıkmeşe (2018), Malyuta et al. (2022) IEEE CSM.
//!
//! # SCvx Subproblem Formulation
//!
//! ```text
//! At iteration j with reference trajectory (x̄, ū):
//!
//! min  -z_N + λ·Σ‖ν_k‖₁          (fuel + virtual control penalty)
//! s.t.
//!   x_{k+1} = x_k + dt·(f̄_k + A_k(x_k-x̄_k) + B_k(u_k-ū_k)) + ν_k
//!   [boundary conditions: x₀ fixed, r_N=0, v_N=0]
//!   [thrust: ‖T_k‖₂ ≤ σ_k, σ ∈ [0, μ₂·(1-(z-z₀))]]
//!   [trust region: ‖x_k - x̄_k‖₂ ≤ r_tr]
//!
//! where ν_k is virtual control (slack for linearization error),
//!       λ = 1e3 penalty drives ν→0 at convergence.
//!
//! L1 virtual control via slack: |ν_j| ≤ s_j ⟺ s_j-ν_j ≥ 0 ∧ s_j+ν_j ≥ 0
//! ```
//!
//! Variable layout:
//!   [states NX*(N+1) | controls NU*N | virtual NVC*N | vc_slack NVC*N]
//!
//! Cone ordering:
//!   ZeroCone → NonnegativeCone → SOC(4) thrust → SOC(NX+1) trust region

use crate::guidance::linearize::{DragParams, Linearization, ScvxVehicleParams};
use crate::guidance::ScvxConfig;
use crate::solver::socp::SocpData;
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::*;
use nalgebra::DVector;

struct Triplets {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl Triplets {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }
    fn push(&mut self, r: usize, c: usize, v: f64) {
        self.rows.push(r);
        self.cols.push(c);
        self.vals.push(v);
    }
}

const NX: usize = 14;
const NU: usize = 4;
const NVC: usize = 10;
const VC_STATE_INDICES: [usize; 10] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/// Build the SOCP subproblem for one SCvx iteration.
///
/// # Arguments
/// * `x_ref` - Reference state trajectory, length N+1
/// * `u_ref` - Reference control trajectory, length N
/// * `lin` - Linearization data (A, B, f_ref at each timestep)
/// * `dt` - Time step
/// * `gravity` - Gravitational acceleration magnitude
/// * `_drag` - Drag parameters (currently unused in constraints)
/// * `veh` - Vehicle parameters
/// * `config` - SCvx configuration (trust region radius, etc.)
#[allow(clippy::too_many_arguments)]
pub fn build_scvx_subproblem(
    x_ref: &[DVector<f64>],
    u_ref: &[DVector<f64>],
    lin: &Linearization,
    dt: f64,
    _gravity: f64,
    _drag: &DragParams,
    veh: &ScvxVehicleParams,
    config: &ScvxConfig,
) -> SocpData {
    let n = u_ref.len();
    let m_wet = x_ref[0][13].exp();
    let z0 = x_ref[0][13];
    let z_dry = (m_wet * 0.79).ln(); // ~ln(1505) for Acikemese params
    let mu2 = veh.thrust_max / m_wet;

    // Variable offsets
    let state_off = 0;
    let ctrl_off = NX * (n + 1);
    let vc_off = ctrl_off + NU * n;
    let slack_off = vc_off + NVC * n;
    let nvar = slack_off + NVC * n;

    // Index helpers
    let ix = |k: usize, i: usize| -> usize { state_off + NX * k + i };
    let iu = |k: usize, i: usize| -> usize { ctrl_off + NU * k + i };
    let iv = |k: usize, i: usize| -> usize { vc_off + NVC * k + i };
    let is_fn = |k: usize, i: usize| -> usize { slack_off + NVC * k + i };

    // Count constraints
    let n_eq = NX * n + NX + 6; // dynamics(NX*N) + init BC(NX) + terminal BC(6)
    let n_ineq = 3 * n + 1 + 2 * NVC * n; // thrust_upper(N) + mass_lower(N+1) + sigma_nonneg(N) + vc_slack(2*NVC*N)
    let n_soc_thrust = 4 * n; // N cones of size 4
    let n_soc_tr = (NX + 1) * (n + 1); // (N+1) cones of size NX+1=15
    let total_rows = n_eq + n_ineq + n_soc_thrust + n_soc_tr;

    let mut t = Triplets::new();
    let mut b = vec![0.0f64; total_rows];
    let mut row = 0;

    // =========================================================================
    // ZeroCone: linearized dynamics
    // x_{k+1} = x_k + dt*(f_ref + A*(x_k - x_ref) + B*(u_k - u_ref)) + vc_insert*nu_k
    // Rewritten for Ax+s=b form (s=0 for equality):
    //   x_{k+1} - (I + dt*A_k)*x_k - dt*B_k*u_k - vc_insert*nu_k = c_k
    // where c_k = dt*(f_ref - A*x_ref - B*u_ref)
    // =========================================================================
    for k in 0..n {
        let ak = &lin.a[k];
        let bk = &lin.b[k];
        let fk = &lin.f_ref[k];
        let c_k = dt * (fk - ak * &x_ref[k] - bk * &u_ref[k]);

        for i in 0..NX {
            // x_{k+1}[i]
            t.push(row, ix(k + 1, i), 1.0);
            // -(I + dt*A_k)[i,:] * x_k
            for j in 0..NX {
                let val = if i == j { -1.0 } else { 0.0 } - dt * ak[(i, j)];
                if val.abs() > 1e-15 {
                    t.push(row, ix(k, j), val);
                }
            }
            // -dt*B_k[i,:]*u_k
            for j in 0..NU {
                let val = -dt * bk[(i, j)];
                if val.abs() > 1e-15 {
                    t.push(row, iu(k, j), val);
                }
            }
            // Virtual control: -nu_{k,vi} for states in VC_STATE_INDICES
            if let Some(vi) = VC_STATE_INDICES.iter().position(|&si| si == i) {
                t.push(row, iv(k, vi), -1.0);
            }
            b[row] = c_k[i];
            row += 1;
        }
    }

    // =========================================================================
    // ZeroCone: Initial boundary condition (all 14 states fixed)
    // =========================================================================
    for i in 0..NX {
        t.push(row, ix(0, i), 1.0);
        b[row] = x_ref[0][i];
        row += 1;
    }

    // =========================================================================
    // ZeroCone: Terminal boundary condition (r=0, v=0, 6 equations)
    // =========================================================================
    for i in 0..6 {
        t.push(row, ix(n, i), 1.0);
        b[row] = 0.0;
        row += 1;
    }
    assert_eq!(row, n_eq, "Equality constraint count mismatch");

    // =========================================================================
    // NonnegativeCone: s = b - A*x >= 0
    // =========================================================================

    // Thrust upper: mu2*(1+z0) - sigma - mu2*z >= 0
    // => -sigma - mu2*z + s = mu2*(1+z0)  =>  A: sigma -> 1, z -> mu2
    for k in 0..n {
        t.push(row, iu(k, 3), 1.0); // sigma coefficient (subtracted by Clarabel: s = b - Ax)
        t.push(row, ix(k, 13), mu2); // z coefficient
        b[row] = mu2 * (1.0 + z0);
        row += 1;
    }

    // Mass lower: z_k - z_dry >= 0
    // => s = z_k - z_dry  =>  -z_k + s = -z_dry  =>  A: z -> -1, b = -z_dry
    for k in 0..=n {
        t.push(row, ix(k, 13), -1.0);
        b[row] = -z_dry;
        row += 1;
    }

    // Sigma nonneg: sigma_k >= 0
    // => s = sigma_k  =>  -sigma_k + s = 0  =>  A: sigma -> -1, b = 0
    for k in 0..n {
        t.push(row, iu(k, 3), -1.0);
        b[row] = 0.0;
        row += 1;
    }

    // VC slack L1: s_{k,j} - nu_{k,j} >= 0  AND  s_{k,j} + nu_{k,j} >= 0
    // First: s_slack = s_{k,j} - nu_{k,j}  =>  -s_{k,j} + nu_{k,j} + s_slack = 0
    //   A: s_{k,j} -> -1, nu_{k,j} -> 1
    // Second: s_slack = s_{k,j} + nu_{k,j}  =>  -s_{k,j} - nu_{k,j} + s_slack = 0
    //   A: s_{k,j} -> -1, nu_{k,j} -> -1
    for k in 0..n {
        for j in 0..NVC {
            // s_{k,j} - nu_{k,j} >= 0
            t.push(row, is_fn(k, j), -1.0);
            t.push(row, iv(k, j), 1.0);
            b[row] = 0.0;
            row += 1;
            // s_{k,j} + nu_{k,j} >= 0
            t.push(row, is_fn(k, j), -1.0);
            t.push(row, iv(k, j), -1.0);
            b[row] = 0.0;
            row += 1;
        }
    }
    assert_eq!(row, n_eq + n_ineq, "Inequality constraint count mismatch");

    // =========================================================================
    // SecondOrderCone(4): Thrust ||T_k|| <= sigma_k, N cones
    // s = [sigma_k; T1; T2; T3], s = b - A*x
    // s[0] = sigma => A: sigma -> -1
    // s[1] = T1    => A: T1 -> -1
    // etc.
    // =========================================================================
    for k in 0..n {
        // s[0] = sigma_k
        t.push(row, iu(k, 3), -1.0);
        b[row] = 0.0;
        row += 1;
        // s[1..3] = T_k
        for j in 0..3 {
            t.push(row, iu(k, j), -1.0);
            b[row] = 0.0;
            row += 1;
        }
    }

    // =========================================================================
    // SecondOrderCone(NX+1=15): Trust region ||x_k - x_ref_k|| <= r_tr
    // s = [r_tr; x_k[0]-x_ref[0]; ...; x_k[13]-x_ref[13]]
    // s[0] = r_tr (constant)         => b[row] = r_tr, no A entry
    // s[i+1] = x_k[i] - x_ref_k[i]  => A[row, ix(k,i)] = -1.0, b[row] = -x_ref_k[i]
    // =========================================================================
    let r_tr = config.trust_region_radius;
    for k in 0..=n {
        // s[0] = r_tr
        b[row] = r_tr;
        row += 1;
        // s[1..NX] = x_k - x_ref_k
        for i in 0..NX {
            t.push(row, ix(k, i), -1.0);
            b[row] = -x_ref[k][i];
            row += 1;
        }
    }
    assert_eq!(row, total_rows, "Total constraint row count mismatch");

    // =========================================================================
    // Objective: min -z_N + lambda * sum(s_{k,j})
    // =========================================================================
    let p = CscMatrix::zeros((nvar, nvar));
    let mut q_obj = vec![0.0; nvar];
    q_obj[ix(n, 13)] = -1.0; // -z_N (maximize final mass)
    let lambda = 1e3;
    for k in 0..n {
        for j in 0..NVC {
            q_obj[is_fn(k, j)] = lambda;
        }
    }

    // =========================================================================
    // Cones
    // =========================================================================
    let mut cones = Vec::new();
    cones.push(ZeroConeT(n_eq));
    cones.push(NonnegativeConeT(n_ineq));
    for _ in 0..n {
        cones.push(SecondOrderConeT(4));
    }
    for _ in 0..=n {
        cones.push(SecondOrderConeT(NX + 1));
    }

    let a_mat = CscMatrix::new_from_triplets(total_rows, nvar, t.rows, t.cols, t.vals);
    SocpData {
        p,
        q: q_obj,
        a: a_mat,
        b,
        cones,
    }
}
