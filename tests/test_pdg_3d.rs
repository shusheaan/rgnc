//! Validation tests for the 3D PDG SOCP solver.
//!
//! Phase 5: Verify 3D formulation is consistent with 2D when y=0.
//! Uses same Açıkmeşe & Ploen (2007) parameters as test_pdg_socp.rs.

use nalgebra::{UnitQuaternion, Vector3};
use rgnc::dynamics::State;
use rgnc::guidance::{
    solve_pdg_2d, solve_pdg_3d, solve_pdg_3d_free_tf, PdgProblem, SolveStatus,
};
use rgnc::vehicle::VehicleParams;

fn acikemese_vehicle() -> VehicleParams {
    VehicleParams {
        dry_mass: 1505.0,
        fuel_mass: 400.0,
        thrust_max: 18600.0,
        thrust_min: 8400.0,
        isp: 225.0,
        ref_area: 1.0,
        glideslope_angle: 4.0_f64.to_radians(),
        inertia: Vector3::new(1000.0, 1000.0, 200.0),
        ref_length: 3.0,
        cg_offset: Vector3::zeros(),
        cp_offset: Vector3::new(2.0, 0.0, 0.0),
        engine_offset: Vector3::new(-3.0, 0.0, 0.0),
        gimbal_max: 0.2618,
    }
}

fn problem_3d(n: usize, dt: f64, y0: f64, vy0: f64) -> PdgProblem {
    let id = UnitQuaternion::identity();
    PdgProblem {
        vehicle: acikemese_vehicle(),
        initial_state: State::new(
            Vector3::new(2000.0, y0, 1500.0),
            Vector3::new(-75.0, vy0, 100.0),
            id, Vector3::zeros(), 1905.0, 0.0,
        ),
        target_state: State::new(
            Vector3::zeros(), Vector3::zeros(),
            id, Vector3::zeros(), 1505.0, 0.0,
        ),
        n_timesteps: n,
        dt,
        gravity: 3.7114,
    }
}

fn problem_2d(n: usize, dt: f64) -> PdgProblem {
    let id = UnitQuaternion::identity();
    PdgProblem {
        vehicle: acikemese_vehicle(),
        initial_state: State::new(
            Vector3::new(2000.0, 0.0, 1500.0),
            Vector3::new(-75.0, 0.0, 100.0),
            id, Vector3::zeros(), 1905.0, 0.0,
        ),
        target_state: State::new(
            Vector3::zeros(), Vector3::zeros(),
            id, Vector3::zeros(), 1505.0, 0.0,
        ),
        n_timesteps: n,
        dt,
        gravity: 3.7114,
    }
}

// === 3D solver returns Solved ===
#[test]
fn test_3d_solver_status() {
    let sol = solve_pdg_3d(&problem_3d(50, 1.5, 0.0, 0.0), false);
    assert_eq!(sol.status, SolveStatus::Optimal);
}

// === 3D terminal conditions ===
#[test]
fn test_3d_terminal_conditions() {
    let sol = solve_pdg_3d(&problem_3d(50, 1.5, 0.0, 0.0), false);
    assert_eq!(sol.status, SolveStatus::Optimal);
    let fin = sol.trajectory.last().unwrap();
    assert!(fin.pos.norm() < 0.01, "pos_err={:.2e}", fin.pos.norm());
    assert!(fin.vel.norm() < 0.01, "vel_err={:.2e}", fin.vel.norm());
}

// === KEY TEST: 3D with y=0 matches 2D ===
#[test]
fn test_3d_matches_2d_when_y_zero() {
    let sol_2d = solve_pdg_2d(&problem_2d(50, 1.5), false);
    let sol_3d = solve_pdg_3d(&problem_3d(50, 1.5, 0.0, 0.0), false);

    assert_eq!(sol_2d.status, SolveStatus::Optimal);
    assert_eq!(sol_3d.status, SolveStatus::Optimal);

    // Fuel should match closely. With glideslope, 3D SOC(3) cone is slightly
    // different from 2D SOC(2) even when y=0, so we test both with and without gs.
    let fuel_diff = (sol_3d.fuel_used - sol_2d.fuel_used).abs();
    let fuel_pct = fuel_diff / sol_2d.fuel_used * 100.0;
    eprintln!("  2D fuel (w/ gs): {:.4} kg", sol_2d.fuel_used);
    eprintln!("  3D fuel (w/ gs): {:.4} kg", sol_3d.fuel_used);
    eprintln!("  diff (w/ gs):    {:.4} kg ({:.4}%)", fuel_diff, fuel_pct);

    // Without glideslope: should match much tighter
    let mut p2_nogs = problem_2d(50, 1.5);
    p2_nogs.vehicle.glideslope_angle = 0.0;
    let mut p3_nogs = problem_3d(50, 1.5, 0.0, 0.0);
    p3_nogs.vehicle.glideslope_angle = 0.0;
    let sol_2d_nogs = solve_pdg_2d(&p2_nogs, false);
    let sol_3d_nogs = solve_pdg_3d(&p3_nogs, false);
    let diff_nogs = (sol_3d_nogs.fuel_used - sol_2d_nogs.fuel_used).abs();
    let pct_nogs = diff_nogs / sol_2d_nogs.fuel_used * 100.0;
    eprintln!("  2D fuel (no gs): {:.4} kg", sol_2d_nogs.fuel_used);
    eprintln!("  3D fuel (no gs): {:.4} kg", sol_3d_nogs.fuel_used);
    eprintln!("  diff (no gs):    {:.4} kg ({:.4}%)", diff_nogs, pct_nogs);

    // Without glideslope: must match within 0.1%
    assert!(
        pct_nogs < 0.1,
        "fuel mismatch (no gs): 2D={:.4}, 3D={:.4}, diff={:.4}%",
        sol_2d_nogs.fuel_used, sol_3d_nogs.fuel_used, pct_nogs,
    );
    // With glideslope: allow up to 2% (SOC dimension difference)
    assert!(
        fuel_pct < 2.0,
        "fuel mismatch (w/ gs): 2D={:.4}, 3D={:.4}, diff={:.4}%",
        sol_2d.fuel_used, sol_3d.fuel_used, fuel_pct,
    );

    // Without glideslope: y=0 throughout, and trajectories match exactly
    for (k, s) in sol_3d_nogs.trajectory.iter().enumerate() {
        assert!(s.pos.y.abs() < 0.01, "r_y[{}]={:.4} should be ~0", k, s.pos.y);
        assert!(s.vel.y.abs() < 0.01, "v_y[{}]={:.4} should be ~0", k, s.vel.y);
    }
    for k in 0..=50 {
        let s2 = &sol_2d_nogs.trajectory[k];
        let s3 = &sol_3d_nogs.trajectory[k];
        let pos_diff = ((s2.pos.x - s3.pos.x).powi(2) + (s2.pos.z - s3.pos.z).powi(2)).sqrt();
        assert!(
            pos_diff < 0.1,
            "position mismatch (no gs) at k={}: 2D=({:.4},{:.4}), 3D=({:.4},{:.4}), diff={:.4}",
            k, s2.pos.x, s2.pos.z, s3.pos.x, s3.pos.z, pos_diff,
        );
    }
}

// === 3D with nonzero y: still feasible and uses more fuel ===
#[test]
fn test_3d_with_crossrange() {
    let sol_base = solve_pdg_3d(&problem_3d(50, 1.5, 0.0, 0.0), false);
    let sol_cross = solve_pdg_3d(&problem_3d(50, 1.5, 500.0, -30.0), false);

    assert_eq!(sol_base.status, SolveStatus::Optimal);
    assert_eq!(sol_cross.status, SolveStatus::Optimal);

    // More initial displacement → more fuel
    eprintln!("  base fuel: {:.2} kg", sol_base.fuel_used);
    eprintln!("  crossrange fuel: {:.2} kg", sol_cross.fuel_used);
    assert!(
        sol_cross.fuel_used > sol_base.fuel_used - 1.0,
        "crossrange should need more or equal fuel",
    );

    // Terminal conditions still met
    let fin = sol_cross.trajectory.last().unwrap();
    assert!(fin.pos.norm() < 0.01);
    assert!(fin.vel.norm() < 0.01);
}

// === 3D glideslope (cone constraint in 3D) ===
#[test]
fn test_3d_glideslope() {
    let sol = solve_pdg_3d(&problem_3d(50, 1.5, 500.0, -30.0), false);
    assert_eq!(sol.status, SolveStatus::Optimal);

    let tan_gs = 4.0_f64.to_radians().tan();
    for (k, s) in sol.trajectory.iter().enumerate() {
        let r_horiz = (s.pos.x.powi(2) + s.pos.y.powi(2)).sqrt();
        assert!(
            s.pos.z >= tan_gs * r_horiz - 0.01,
            "glideslope k={}: alt={:.2} < tan(4°)*{:.2}={:.2}",
            k, s.pos.z, r_horiz, tan_gs * r_horiz,
        );
    }
}

// === 3D mass bounds ===
#[test]
fn test_3d_mass_bounds() {
    let sol = solve_pdg_3d(&problem_3d(50, 1.5, 0.0, 0.0), false);
    assert_eq!(sol.status, SolveStatus::Optimal);
    for (k, s) in sol.trajectory.iter().enumerate() {
        assert!(s.mass >= 1505.0 - 0.1, "m[{}]={:.2} < m_dry", k, s.mass);
    }
    // Monotone
    for k in 1..sol.trajectory.len() {
        assert!(sol.trajectory[k].mass <= sol.trajectory[k - 1].mass + 0.01);
    }
}

// === 3D free-tf optimization ===
#[test]
fn test_3d_free_tf() {
    let fixed = solve_pdg_3d(&problem_3d(50, 1.5, 0.0, 0.0), false);
    let free = solve_pdg_3d_free_tf(&problem_3d(50, 1.0, 0.0, 0.0), (0.5, 3.0), false);

    assert_eq!(fixed.status, SolveStatus::Optimal);
    assert_eq!(free.status, SolveStatus::Optimal);

    eprintln!("  fixed tf fuel: {:.2} kg", fixed.fuel_used);
    eprintln!("  free  tf fuel: {:.2} kg", free.fuel_used);

    assert!(
        free.fuel_used <= fixed.fuel_used + 1.0,
        "free_tf should be no worse",
    );
}

// === 3D dynamics defect ===
#[test]
fn test_3d_dynamics_defect() {
    let prob = problem_3d(50, 1.5, 0.0, 0.0);
    let raw = rgnc::guidance::formulation::build_pdg_3d(&prob);
    let result = rgnc::solver::socp::solve_socp(&raw, false);
    assert_eq!(result.status, rgnc::solver::result::SolveStatusGeneric::Solved);

    let x = &result.primal;
    let n = 50;
    let dt = 1.5;
    let dt2 = dt / 2.0;
    let g = 3.7114;
    let ctrl_off = 6 * (n + 1);

    let mut max_pos = 0.0f64;
    let mut max_vel = 0.0f64;
    for k in 0..n {
        let u1 = x[ctrl_off + 4 * k];
        let u2 = x[ctrl_off + 4 * k + 1];
        let u3 = x[ctrl_off + 4 * k + 2];

        // Trapezoidal position
        for d in 0..3 {
            let rk = x[6 * k + d];
            let rk1 = x[6 * (k + 1) + d];
            let vk = x[6 * k + 3 + d];
            let vk1 = x[6 * (k + 1) + 3 + d];
            let re = rk + dt2 * (vk + vk1);
            max_pos = max_pos.max((re - rk1).abs());
        }

        // Euler velocity
        let u = [u1, u2, u3];
        let grav = [0.0, 0.0, -g];
        for d in 0..3 {
            let vk = x[6 * k + 3 + d];
            let vk1 = x[6 * (k + 1) + 3 + d];
            let ve = vk + dt * (u[d] + grav[d]);
            max_vel = max_vel.max((ve - vk1).abs());
        }
    }
    eprintln!("  3D pos defect: {:.2e}", max_pos);
    eprintln!("  3D vel defect: {:.2e}", max_vel);
    assert!(max_pos < 1e-8, "pos defect {:.2e}", max_pos);
    assert!(max_vel < 1e-8, "vel defect {:.2e}", max_vel);
}
