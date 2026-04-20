//! SOCP matrix construction for Powered Descent Guidance (PDG).
//!
//! Implements Açıkmeşe & Ploen (2007) "Convex Programming Approach to
//! Powered Descent Guidance for Mars Landing", JGCD 30(5).
//!
//! # Mathematical Formulation
//!
//! ```text
//! Decision variables per timestep k:
//!   r_k ∈ R^d (position), v_k ∈ R^d (velocity),
//!   u_k ∈ R^d (thrust accel), σ_k ∈ R (thrust slack), z_k ∈ R (log-mass)
//!
//! Objective: min -z_N  (maximize terminal mass = minimize fuel)
//!
//! Dynamics (trapezoidal position, Euler velocity):
//!   r_{k+1} = r_k + (dt/2)(v_k + v_{k+1})
//!   v_{k+1} = v_k + dt(g + u_k)
//!   z_{k+1} = z_k - α·dt·σ_k         where α = 1/(Isp·g₀)
//!
//! Lossless convexification (Açıkmeşe 2007, Thm 1):
//!   Relaxes T ∈ {0}∪[T_min,T_max] to T ∈ [0,T_max].
//!   Optimal solution automatically avoids the gap (0,T_min).
//!   ‖u_k‖₂ ≤ σ_k                     (SOC: thrust direction)
//!   σ_k ≤ μ₂·(1 - (z_k - z₀))       (linearized upper bound)
//!   σ_k ≥ 0                           (nonneg)
//!   where μ₂ = T_max/m_wet
//!
//! Glideslope: r_alt ≥ tan(γ_gs)·‖r_horiz‖₂  (SOC)
//!
//! Boundary: r₀=r_init, v₀=v_init, z₀=ln(m_wet), r_N=0, v_N=0
//! ```
//!
//! # Clarabel Mapping
//!
//! Standard form: min (1/2)x'Px + q'x, s.t. Ax + s = b, s ∈ K
//!
//! Variable layout: [states | controls | log-mass]
//!   2D: states = (r1,r2,v1,v2) per k, controls = (u1,u2,σ) per k
//!   3D: states = (r1,r2,r3,v1,v2,v3) per k, controls = (u1,u2,u3,σ) per k
//!
//! Cone ordering: ZeroCone → NonnegativeCone → SOC(thrust) → SOC(glideslope)

use crate::guidance::PdgProblem;
use crate::solver::socp::SocpData;
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::*;

/// Variable index calculator for the 2D PDG SOCP.
struct Idx {
    n: usize,
    ctrl_off: usize,
    mass_off: usize,
}

impl Idx {
    fn new(n: usize) -> Self {
        let ctrl_off = 4 * (n + 1);
        let mass_off = ctrl_off + 3 * n;
        Self { n, ctrl_off, mass_off }
    }
    fn total(&self) -> usize { self.mass_off + self.n + 1 }
    fn r1(&self, k: usize) -> usize { 4 * k }
    fn r2(&self, k: usize) -> usize { 4 * k + 1 }
    fn v1(&self, k: usize) -> usize { 4 * k + 2 }
    fn v2(&self, k: usize) -> usize { 4 * k + 3 }
    fn u1(&self, k: usize) -> usize { self.ctrl_off + 3 * k }
    fn u2(&self, k: usize) -> usize { self.ctrl_off + 3 * k + 1 }
    fn sigma(&self, k: usize) -> usize { self.ctrl_off + 3 * k + 2 }
    fn z(&self, k: usize) -> usize { self.mass_off + k }
}

/// Triplet accumulator for building sparse matrices.
struct Triplets {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl Triplets {
    fn new() -> Self {
        Self { rows: Vec::new(), cols: Vec::new(), vals: Vec::new() }
    }
    fn push(&mut self, r: usize, c: usize, v: f64) {
        self.rows.push(r);
        self.cols.push(c);
        self.vals.push(v);
    }
}

/// Build the 2D PDG SOCP data for Clarabel.
///
/// Implements Açıkmeşe & Ploen (2007) formulation:
/// - Euler discretization
/// - Log-mass z = ln(m) with linearized thrust bounds
/// - Glideslope as second-order cone constraint
pub fn build_pdg_2d(problem: &PdgProblem) -> SocpData {
    let n = problem.n_timesteps;
    let dt = problem.dt;
    let v = &problem.vehicle;
    let g = problem.gravity;
    let idx = Idx::new(n);
    let nvar = idx.total();

    let m_wet = v.dry_mass + v.fuel_mass;
    let alpha = 1.0 / (v.isp * 9.80665);
    let z0 = m_wet.ln();
    let z_dry = v.dry_mass.ln();
    let _mu1 = v.thrust_min / m_wet; // kept for future SCvx lower bound re-addition
    let mu2 = v.thrust_max / m_wet;
    let tan_gs = v.glideslope_angle.tan();
    let has_gs = tan_gs > 1e-10;

    // 2D: r1=downrange(x), r2=altitude(z)
    let r1_init = problem.initial_state.pos.x;
    let r2_init = problem.initial_state.pos.z;
    let v1_init = problem.initial_state.vel.x;
    let v2_init = problem.initial_state.vel.z;
    let r1_target = problem.target_state.pos.x;
    let r2_target = problem.target_state.pos.z;
    let v1_target = problem.target_state.vel.x;
    let v2_target = problem.target_state.vel.z;

    // --- Count constraint rows ---
    let n_eq = 5 * n + 9; // dynamics(5*N) + BC_init(5) + BC_final(4)
    // Lossless convexification: relax σ ∈ {0}∪[T_min/m, T_max/m] to 0 ≤ σ ≤ T_max/m
    // So we only need upper thrust bound + mass lower bound + sigma nonneg
    let n_ineq = 2 * n + (n + 1); // thrust_upper(N) + sigma_nonneg(N) + mass_bounds(N+1)
    let n_soc_thrust = 3 * n; // N cones of dim 3
    let n_soc_gs = if has_gs { 2 * (n + 1) } else { 0 };
    let total_rows = n_eq + n_ineq + n_soc_thrust + n_soc_gs;

    let mut t = Triplets::new();
    let mut b = vec![0.0f64; total_rows];
    let mut row = 0;

    // ============================
    // ZeroCone: equalities
    // ============================

    // Position dynamics (trapezoidal): r_{k+1} = r_k + dt/2 * (v_k + v_{k+1})
    let dt2 = dt / 2.0;
    for k in 0..n {
        t.push(row, idx.r1(k), -1.0);
        t.push(row, idx.r1(k + 1), 1.0);
        t.push(row, idx.v1(k), -dt2);
        t.push(row, idx.v1(k + 1), -dt2);
        row += 1;

        t.push(row, idx.r2(k), -1.0);
        t.push(row, idx.r2(k + 1), 1.0);
        t.push(row, idx.v2(k), -dt2);
        t.push(row, idx.v2(k + 1), -dt2);
        row += 1;
    }

    // Velocity dynamics: v_{k+1} = v_k + dt*(gravity + u_k)
    // gravity = [0, -g] where g is positive magnitude
    for k in 0..n {
        t.push(row, idx.v1(k), -1.0);
        t.push(row, idx.v1(k + 1), 1.0);
        t.push(row, idx.u1(k), -dt);
        b[row] = 0.0;
        row += 1;

        t.push(row, idx.v2(k), -1.0);
        t.push(row, idx.v2(k + 1), 1.0);
        t.push(row, idx.u2(k), -dt);
        b[row] = -dt * g; // gravity pulls down
        row += 1;
    }

    // Mass dynamics: z_{k+1} = z_k - α*dt*σ_k
    for k in 0..n {
        t.push(row, idx.z(k), -1.0);
        t.push(row, idx.z(k + 1), 1.0);
        t.push(row, idx.sigma(k), alpha * dt);
        row += 1;
    }

    // Initial boundary conditions
    t.push(row, idx.r1(0), 1.0);
    b[row] = r1_init;
    row += 1;
    t.push(row, idx.r2(0), 1.0);
    b[row] = r2_init;
    row += 1;
    t.push(row, idx.v1(0), 1.0);
    b[row] = v1_init;
    row += 1;
    t.push(row, idx.v2(0), 1.0);
    b[row] = v2_init;
    row += 1;
    t.push(row, idx.z(0), 1.0);
    b[row] = z0;
    row += 1;

    // Terminal boundary conditions
    t.push(row, idx.r1(n), 1.0);
    b[row] = r1_target;
    row += 1;
    t.push(row, idx.r2(n), 1.0);
    b[row] = r2_target;
    row += 1;
    t.push(row, idx.v1(n), 1.0);
    b[row] = v1_target;
    row += 1;
    t.push(row, idx.v2(n), 1.0);
    b[row] = v2_target;
    row += 1;

    assert_eq!(row, n_eq);

    // ============================
    // NonnegativeCone: inequalities (s = b - Ax ≥ 0)
    // ============================

    // Thrust upper bound (lossless convexification removes lower bound): σ_k ≤ μ₂*(1 - (z_k - z₀))
    //   μ₂*(1+z₀) - σ_k - μ₂*z_k ≥ 0
    //   A: σ_k → 1, z_k → μ₂; b = μ₂*(1+z₀)
    for k in 0..n {
        t.push(row, idx.sigma(k), 1.0);
        t.push(row, idx.z(k), mu2);
        b[row] = mu2 * (1.0 + z0);
        row += 1;
    }

    // Mass lower bound: z_k ≥ ln(m_dry)
    //   z_k - z_dry ≥ 0  →  A: z_k → -1; b = -z_dry
    for k in 0..=n {
        t.push(row, idx.z(k), -1.0);
        b[row] = -z_dry;
        row += 1;
    }

    // σ_k ≥ 0 (thrust magnitude is nonneg)
    //   A: σ_k → -1; b = 0
    for k in 0..n {
        t.push(row, idx.sigma(k), -1.0);
        b[row] = 0.0;
        row += 1;
    }

    assert_eq!(row, n_eq + n_ineq);

    // ============================
    // SOC: thrust magnitude ||u_k|| ≤ σ_k
    // ============================
    // Clarabel SOC(3): s[0] ≥ ||(s[1], s[2])||
    // s = b - Ax, so A_row = -1 at the variable, b = 0
    for k in 0..n {
        t.push(row, idx.sigma(k), -1.0);
        row += 1;
        t.push(row, idx.u1(k), -1.0);
        row += 1;
        t.push(row, idx.u2(k), -1.0);
        row += 1;
    }

    // ============================
    // SOC: glideslope r2_k ≥ tan(γ)*|r1_k|
    // ============================
    // SOC(2): [r2_k/tan(γ), r1_k] → r2_k/tan(γ) ≥ |r1_k|
    // s[0] = -(-1/tan_gs)*r2_k = r2_k/tan_gs
    // s[1] = -(-1)*r1_k = r1_k
    if has_gs {
        for k in 0..=n {
            t.push(row, idx.r2(k), -1.0 / tan_gs);
            row += 1;
            t.push(row, idx.r1(k), -1.0);
            row += 1;
        }
    }

    assert_eq!(row, total_rows);

    // ============================
    // Objective: min -z_N (maximize final mass)
    // ============================
    let p = CscMatrix::zeros((nvar, nvar));
    let mut q = vec![0.0; nvar];
    q[idx.z(n)] = -1.0;

    // ============================
    // Cone specification
    // ============================
    let mut cones = Vec::new();
    cones.push(ZeroConeT(n_eq));
    cones.push(NonnegativeConeT(n_ineq));
    for _ in 0..n {
        cones.push(SecondOrderConeT(3));
    }
    if has_gs {
        for _ in 0..=n {
            cones.push(SecondOrderConeT(2));
        }
    }

    // Build A from triplets
    let a = CscMatrix::new_from_triplets(total_rows, nvar, t.rows, t.cols, t.vals);

    SocpData { p, q, a, b, cones }
}

// ===========================================================
// 3D PDG formulation
// ===========================================================

/// Variable index calculator for the 3D PDG SOCP.
struct Idx3 {
    n: usize,
    ctrl_off: usize,
    mass_off: usize,
}

impl Idx3 {
    fn new(n: usize) -> Self {
        let ctrl_off = 6 * (n + 1);       // 6 state vars per timestep
        let mass_off = ctrl_off + 4 * n;   // 4 control vars per timestep
        Self { n, ctrl_off, mass_off }
    }
    fn total(&self) -> usize { self.mass_off + self.n + 1 }
    fn r1(&self, k: usize) -> usize { 6 * k }
    fn r2(&self, k: usize) -> usize { 6 * k + 1 }
    fn r3(&self, k: usize) -> usize { 6 * k + 2 }
    fn v1(&self, k: usize) -> usize { 6 * k + 3 }
    fn v2(&self, k: usize) -> usize { 6 * k + 4 }
    fn v3(&self, k: usize) -> usize { 6 * k + 5 }
    fn u1(&self, k: usize) -> usize { self.ctrl_off + 4 * k }
    fn u2(&self, k: usize) -> usize { self.ctrl_off + 4 * k + 1 }
    fn u3(&self, k: usize) -> usize { self.ctrl_off + 4 * k + 2 }
    fn sigma(&self, k: usize) -> usize { self.ctrl_off + 4 * k + 3 }
    fn z(&self, k: usize) -> usize { self.mass_off + k }
}

/// Build the 3D PDG SOCP data for Clarabel.
///
/// Extends the 2D formulation to 3 spatial dimensions:
/// - State: [r1, r2, r3, v1, v2, v3] per timestep (r3 = altitude)
/// - Control: [u1, u2, u3, σ] per timestep
/// - Glideslope: r3 ≥ tan(γ) · ||(r1, r2)|| → SOC(3)
/// - Thrust: ||(u1, u2, u3)|| ≤ σ → SOC(4)
pub fn build_pdg_3d(problem: &PdgProblem) -> SocpData {
    let n = problem.n_timesteps;
    let dt = problem.dt;
    let v = &problem.vehicle;
    let g = problem.gravity;
    let idx = Idx3::new(n);
    let nvar = idx.total();

    let m_wet = v.dry_mass + v.fuel_mass;
    let alpha = 1.0 / (v.isp * 9.80665);
    let z0 = m_wet.ln();
    let z_dry = v.dry_mass.ln();
    let mu2 = v.thrust_max / m_wet;
    let tan_gs = v.glideslope_angle.tan();
    let has_gs = tan_gs > 1e-10;

    // 3D: r1=x(East), r2=y(North), r3=z(Up/altitude)
    let r_init = problem.initial_state.pos;
    let v_init = problem.initial_state.vel;
    let r_target = problem.target_state.pos;
    let v_target = problem.target_state.vel;

    // --- Count constraint rows ---
    // Dynamics: 3 position + 3 velocity + 1 mass = 7 per interval → 7*N
    // BC init: 3 pos + 3 vel + 1 mass = 7
    // BC terminal: 3 pos + 3 vel = 6
    let n_eq = 7 * n + 13;
    // Thrust upper: N, mass lower: N+1, sigma nonneg: N
    let n_ineq = 2 * n + (n + 1);
    // Thrust SOC(4): N cones
    let n_soc_thrust = 4 * n;
    // Glideslope SOC(3): N+1 cones (r3/tan_gs ≥ ||(r1, r2)||)
    let n_soc_gs = if has_gs { 3 * (n + 1) } else { 0 };
    let total_rows = n_eq + n_ineq + n_soc_thrust + n_soc_gs;

    let mut t = Triplets::new();
    let mut b = vec![0.0f64; total_rows];
    let mut row = 0;

    // ============================
    // ZeroCone: equalities
    // ============================
    let dt2 = dt / 2.0;

    // Position dynamics (trapezoidal): r_{k+1} = r_k + dt/2 * (v_k + v_{k+1})
    for k in 0..n {
        // r1
        t.push(row, idx.r1(k), -1.0);
        t.push(row, idx.r1(k + 1), 1.0);
        t.push(row, idx.v1(k), -dt2);
        t.push(row, idx.v1(k + 1), -dt2);
        row += 1;
        // r2
        t.push(row, idx.r2(k), -1.0);
        t.push(row, idx.r2(k + 1), 1.0);
        t.push(row, idx.v2(k), -dt2);
        t.push(row, idx.v2(k + 1), -dt2);
        row += 1;
        // r3
        t.push(row, idx.r3(k), -1.0);
        t.push(row, idx.r3(k + 1), 1.0);
        t.push(row, idx.v3(k), -dt2);
        t.push(row, idx.v3(k + 1), -dt2);
        row += 1;
    }

    // Velocity dynamics (Euler): v_{k+1} = v_k + dt*(u_k + gravity)
    // gravity = [0, 0, -g]
    for k in 0..n {
        // v1
        t.push(row, idx.v1(k), -1.0);
        t.push(row, idx.v1(k + 1), 1.0);
        t.push(row, idx.u1(k), -dt);
        row += 1;
        // v2
        t.push(row, idx.v2(k), -1.0);
        t.push(row, idx.v2(k + 1), 1.0);
        t.push(row, idx.u2(k), -dt);
        row += 1;
        // v3 (with gravity)
        t.push(row, idx.v3(k), -1.0);
        t.push(row, idx.v3(k + 1), 1.0);
        t.push(row, idx.u3(k), -dt);
        b[row] = -dt * g;
        row += 1;
    }

    // Mass dynamics: z_{k+1} = z_k - α*dt*σ_k
    for k in 0..n {
        t.push(row, idx.z(k), -1.0);
        t.push(row, idx.z(k + 1), 1.0);
        t.push(row, idx.sigma(k), alpha * dt);
        row += 1;
    }

    // Initial boundary conditions (7: r1,r2,r3, v1,v2,v3, z)
    t.push(row, idx.r1(0), 1.0); b[row] = r_init.x; row += 1;
    t.push(row, idx.r2(0), 1.0); b[row] = r_init.y; row += 1;
    t.push(row, idx.r3(0), 1.0); b[row] = r_init.z; row += 1;
    t.push(row, idx.v1(0), 1.0); b[row] = v_init.x; row += 1;
    t.push(row, idx.v2(0), 1.0); b[row] = v_init.y; row += 1;
    t.push(row, idx.v3(0), 1.0); b[row] = v_init.z; row += 1;
    t.push(row, idx.z(0), 1.0);  b[row] = z0;       row += 1;

    // Terminal boundary conditions (6: r1,r2,r3, v1,v2,v3)
    t.push(row, idx.r1(n), 1.0); b[row] = r_target.x; row += 1;
    t.push(row, idx.r2(n), 1.0); b[row] = r_target.y; row += 1;
    t.push(row, idx.r3(n), 1.0); b[row] = r_target.z; row += 1;
    t.push(row, idx.v1(n), 1.0); b[row] = v_target.x; row += 1;
    t.push(row, idx.v2(n), 1.0); b[row] = v_target.y; row += 1;
    t.push(row, idx.v3(n), 1.0); b[row] = v_target.z; row += 1;

    assert_eq!(row, n_eq);

    // ============================
    // NonnegativeCone: inequalities
    // ============================

    // Thrust upper bound: σ_k ≤ μ₂*(1 - (z_k - z₀))
    for k in 0..n {
        t.push(row, idx.sigma(k), 1.0);
        t.push(row, idx.z(k), mu2);
        b[row] = mu2 * (1.0 + z0);
        row += 1;
    }

    // Mass lower bound: z_k ≥ ln(m_dry)
    for k in 0..=n {
        t.push(row, idx.z(k), -1.0);
        b[row] = -z_dry;
        row += 1;
    }

    // σ_k ≥ 0
    for k in 0..n {
        t.push(row, idx.sigma(k), -1.0);
        row += 1;
    }

    assert_eq!(row, n_eq + n_ineq);

    // ============================
    // SOC: thrust magnitude ||(u1, u2, u3)|| ≤ σ → SOC(4)
    // ============================
    for k in 0..n {
        t.push(row, idx.sigma(k), -1.0); row += 1;
        t.push(row, idx.u1(k), -1.0);    row += 1;
        t.push(row, idx.u2(k), -1.0);    row += 1;
        t.push(row, idx.u3(k), -1.0);    row += 1;
    }

    // ============================
    // SOC: glideslope r3 ≥ tan(γ)*||(r1, r2)|| → SOC(3)
    // ============================
    if has_gs {
        for k in 0..=n {
            t.push(row, idx.r3(k), -1.0 / tan_gs); row += 1;
            t.push(row, idx.r1(k), -1.0);           row += 1;
            t.push(row, idx.r2(k), -1.0);           row += 1;
        }
    }

    assert_eq!(row, total_rows);

    // ============================
    // Objective: min -z_N
    // ============================
    let p = CscMatrix::zeros((nvar, nvar));
    let mut q = vec![0.0; nvar];
    q[idx.z(n)] = -1.0;

    // ============================
    // Cone specification
    // ============================
    let mut cones = Vec::new();
    cones.push(ZeroConeT(n_eq));
    cones.push(NonnegativeConeT(n_ineq));
    for _ in 0..n {
        cones.push(SecondOrderConeT(4)); // 3D thrust
    }
    if has_gs {
        for _ in 0..=n {
            cones.push(SecondOrderConeT(3)); // 3D glideslope
        }
    }

    let a = CscMatrix::new_from_triplets(total_rows, nvar, t.rows, t.cols, t.vals);
    SocpData { p, q, a, b, cones }
}
