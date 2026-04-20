//! Tests for 6-DOF SCvx implementation.

use nalgebra::{DVector, Vector3};
use rgnc::guidance::linearize::{
    scvx_dynamics, compute_jacobians, linearize_trajectory, DragParams, ScvxVehicleParams,
};
use rgnc::guidance::scvx_formulation::build_scvx_subproblem;
use rgnc::guidance::ScvxConfig;
use rgnc::guidance::scvx::{solve_scvx, ScvxProblem};
use rgnc::solver::socp::solve_socp;
use rgnc::solver::result::SolveStatusGeneric;

#[test]
fn test_scvx_dynamics_freefall() {
    let mut x = DVector::zeros(14);
    x[2] = 1000.0;  // altitude
    x[6] = 1.0;     // identity quaternion w
    x[13] = (1905.0_f64).ln();
    let u = DVector::zeros(4);
    let drag = DragParams { ref_area: 0.0, cd: 0.0, density: 0.0 };

    let xdot = scvx_dynamics(&x, &u, 3.7114, &drag);

    assert_eq!(xdot.len(), 14);
    assert!(xdot[0].abs() < 1e-10); // r_dot = 0
    assert!((xdot[5] + 3.7114).abs() < 1e-10, "v3_dot={}", xdot[5]); // v_z_dot = -g
    assert!(xdot[6].abs() < 1e-10); // q_dot = 0
    assert!(xdot[10].abs() < 1e-10); // omega_dot = 0
    assert!(xdot[13].abs() < 1e-10); // z_dot = 0
}

#[test]
fn test_scvx_dynamics_with_thrust() {
    let mut x = DVector::zeros(14);
    x[2] = 1000.0;
    x[6] = 1.0;
    x[13] = (1905.0_f64).ln();
    let mut u = DVector::zeros(4);
    u[2] = 3.7114;  // T_z = g (hover)
    u[3] = 3.7114;  // sigma
    let drag = DragParams { ref_area: 0.0, cd: 0.0, density: 0.0 };

    let xdot = scvx_dynamics(&x, &u, 3.7114, &drag);

    assert!(xdot[5].abs() < 1e-10, "v3_dot={:.2e} not ~0 (hover)", xdot[5]);
    assert!(xdot[13] < 0.0, "z_dot should be negative (burning fuel)");
}

#[test]
fn test_scvx_dynamics_with_drag() {
    let mut x = DVector::zeros(14);
    x[2] = 1000.0;
    x[3] = 100.0;  // v_x = 100 m/s
    x[6] = 1.0;
    x[13] = (1905.0_f64).ln();
    let u = DVector::zeros(4);
    let drag = DragParams { ref_area: 5.0, cd: 1.0, density: 0.02 };

    let xdot = scvx_dynamics(&x, &u, 3.7114, &drag);

    assert!(xdot[3] < 0.0, "drag should decelerate: v1_dot={}", xdot[3]);
    assert!((xdot[5] + 3.7114).abs() < 0.1);
}

#[test]
fn test_jacobian_gravity_only() {
    let mut x = DVector::zeros(14);
    x[2] = 1000.0;
    x[6] = 1.0;
    x[13] = (1905.0_f64).ln();
    let u = DVector::zeros(4);
    let drag = DragParams { ref_area: 0.0, cd: 0.0, density: 0.0 };

    let (a, b) = compute_jacobians(&x, &u, 3.7114, &drag);

    assert_eq!(a.nrows(), 14);
    assert_eq!(a.ncols(), 14);
    assert_eq!(b.nrows(), 14);
    assert_eq!(b.ncols(), 4);

    // dr/dv = I (position depends on velocity)
    for i in 0..3 {
        assert!((a[(i, i + 3)] - 1.0).abs() < 1e-6,
            "dr{}/dv{} = {} expected 1.0", i, i, a[(i, i + 3)]);
    }

    // dv/du: B[3..6, 0..3] = I (thrust accel goes directly to velocity)
    for i in 0..3 {
        assert!((b[(3 + i, i)] - 1.0).abs() < 1e-6,
            "dv{}/dT{} = {} expected 1.0", i, i, b[(3 + i, i)]);
    }

    // dz/dsigma: B[13, 3] = -alpha
    let alpha = 1.0 / (225.0 * 9.80665);
    assert!((b[(13, 3)] + alpha).abs() < 1e-8,
        "dz/dsigma = {} expected {}", b[(13, 3)], -alpha);
}

#[test]
fn test_jacobian_with_drag() {
    let mut x = DVector::zeros(14);
    x[2] = 1000.0;
    x[3] = 100.0;  // v_x = 100 m/s
    x[6] = 1.0;
    x[13] = (1905.0_f64).ln();
    let u = DVector::zeros(4);
    let drag = DragParams { ref_area: 5.0, cd: 1.0, density: 0.02 };

    let (a, _b) = compute_jacobians(&x, &u, 3.7114, &drag);

    // dv_x/dv_x should be negative (drag opposes velocity)
    assert!(a[(3, 3)] < 0.0, "dv1/dv1 = {} should be < 0 (drag)", a[(3, 3)]);
}

fn make_reference_trajectory(n: usize) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
    let r0 = Vector3::new(2000.0, 0.0, 1500.0);
    let v0 = Vector3::new(-75.0, 0.0, 100.0);
    let mut x_ref = Vec::with_capacity(n + 1);
    let mut u_ref = Vec::with_capacity(n);
    let z0 = (1905.0_f64).ln();
    for k in 0..=n {
        let frac = k as f64 / n as f64;
        let mut xk = DVector::zeros(14);
        let r = r0 * (1.0 - frac);
        let v = v0 * (1.0 - frac);
        xk[0] = r.x;
        xk[1] = r.y;
        xk[2] = r.z;
        xk[3] = v.x;
        xk[4] = v.y;
        xk[5] = v.z;
        xk[6] = 1.0; // identity quaternion w
        xk[13] = z0;
        x_ref.push(xk);
    }
    for _ in 0..n {
        let mut uk = DVector::zeros(4);
        uk[2] = 3.7114; // T_z = g (hover thrust)
        uk[3] = 3.7114; // sigma
        u_ref.push(uk);
    }
    (x_ref, u_ref)
}

#[test]
fn test_scvx_subproblem_solvable() {
    let n = 30;
    let drag = DragParams {
        ref_area: 0.0,
        cd: 0.0,
        density: 0.0,
    };
    let veh = ScvxVehicleParams {
        inertia: Vector3::new(1000.0, 1000.0, 200.0),
        isp: 225.0,
        thrust_max: 18600.0,
        cp_offset: Vector3::new(2.0, 0.0, 0.0),
    };
    let (x_ref, u_ref) = make_reference_trajectory(n);
    let lin = linearize_trajectory(&x_ref, &u_ref, 3.7114, &drag, &veh);
    let config = ScvxConfig {
        max_iterations: 50,
        tolerance: 1e-4,
        trust_region_radius: 500.0,
        trust_region_shrink: 0.5,
        trust_region_expand: 2.0,
    };
    let data = build_scvx_subproblem(&x_ref, &u_ref, &lin, 2.0, 3.7114, &drag, &veh, &config);
    let result = solve_socp(&data, false);
    assert_eq!(
        result.status,
        SolveStatusGeneric::Solved,
        "SOCP subproblem should be solvable"
    );
}

#[test]
fn test_scvx_no_drag_converges() {
    let prob = ScvxProblem {
        vehicle: ScvxVehicleParams {
            inertia: Vector3::new(1000.0, 1000.0, 200.0),
            isp: 225.0,
            thrust_max: 18600.0,
            cp_offset: Vector3::new(2.0, 0.0, 0.0),
        },
        drag: DragParams { ref_area: 0.0, cd: 0.0, density: 0.0 },
        gravity: 3.7114,
        initial_pos: Vector3::new(2000.0, 0.0, 1500.0),
        initial_vel: Vector3::new(-75.0, 0.0, 100.0),
        wet_mass: 1905.0,
        dry_mass: 1505.0,
        n_timesteps: 30,
        dt: 2.0,
        config: ScvxConfig {
            max_iterations: 30,
            tolerance: 1e-3,
            trust_region_radius: 500.0,
            trust_region_shrink: 0.5,
            trust_region_expand: 2.0,
        },
    };

    let sol = solve_scvx(&prob);

    eprintln!("  No-drag SCvx: {} iters, fuel={:.2} kg, converged={}",
        sol.iterations, sol.fuel_used, sol.converged);
    for (i, h) in sol.history.iter().enumerate() {
        eprintln!("    iter {}: cost={:.4}, defect_pos={:.2}, defect_vel={:.2}, vc={:.2e}, tr={:.1}",
            i, h.cost, h.defect_pos, h.defect_vel, h.vc_norm, h.trust_radius);
    }

    assert!(sol.converged, "SCvx did not converge in {} iterations", sol.iterations);

    let fin = sol.trajectory.last().unwrap();
    assert!(fin.pos.norm() < 5.0, "terminal pos = {:.2}", fin.pos.norm());
    assert!(fin.vel.norm() < 5.0, "terminal vel = {:.2}", fin.vel.norm());
    assert!(sol.fuel_used > 50.0 && sol.fuel_used < 400.0, "fuel = {:.2}", sol.fuel_used);
}

fn mars_drag_problem() -> ScvxProblem {
    ScvxProblem {
        vehicle: ScvxVehicleParams {
            inertia: Vector3::new(1000.0, 1000.0, 200.0),
            isp: 225.0,
            thrust_max: 18600.0,
            cp_offset: Vector3::new(2.0, 0.0, 0.0),
        },
        drag: DragParams { ref_area: 5.0, cd: 1.0, density: 0.02 },
        gravity: 3.7114,
        initial_pos: Vector3::new(2000.0, 0.0, 1500.0),
        initial_vel: Vector3::new(-75.0, 0.0, 100.0),
        wet_mass: 1905.0,
        dry_mass: 1505.0,
        n_timesteps: 30,
        dt: 2.0,
        config: ScvxConfig {
            max_iterations: 30,
            tolerance: 1e-3,
            trust_region_radius: 500.0,
            trust_region_shrink: 0.5,
            trust_region_expand: 2.0,
        },
    }
}

#[test]
fn test_scvx_mars_drag_converges() {
    let sol = solve_scvx(&mars_drag_problem());

    eprintln!("  Mars+drag SCvx: {} iters, fuel={:.2} kg, converged={}",
        sol.iterations, sol.fuel_used, sol.converged);
    for (i, h) in sol.history.iter().enumerate() {
        eprintln!("    iter {}: cost={:.4}, defect_pos={:.2}, defect_vel={:.2}, vc={:.2e}, tr={:.1}",
            i, h.cost, h.defect_pos, h.defect_vel, h.vc_norm, h.trust_radius);
    }

    assert!(sol.converged, "SCvx did not converge in {} iterations", sol.iterations);
    assert!(sol.iterations <= 20, "Too many iterations: {}", sol.iterations);

    let fin = sol.trajectory.last().unwrap();
    assert!(fin.pos.norm() < 10.0, "terminal pos = {:.2}", fin.pos.norm());
    assert!(fin.vel.norm() < 10.0, "terminal vel = {:.2}", fin.vel.norm());
    assert!(sol.fuel_used > 50.0 && sol.fuel_used < 400.0, "fuel {:.2} unreasonable", sol.fuel_used);
}

#[test]
fn test_scvx_drag_reduces_fuel() {
    let mut prob_nodrag = mars_drag_problem();
    prob_nodrag.drag = DragParams { ref_area: 0.0, cd: 0.0, density: 0.0 };

    let sol_nodrag = solve_scvx(&prob_nodrag);
    let sol_drag = solve_scvx(&mars_drag_problem());

    assert!(sol_nodrag.converged, "no-drag SCvx failed");
    assert!(sol_drag.converged, "drag SCvx failed");

    eprintln!("  no-drag fuel: {:.2} kg", sol_nodrag.fuel_used);
    eprintln!("  w/ drag fuel: {:.2} kg", sol_drag.fuel_used);

    // Drag helps decelerate → less fuel needed
    assert!(
        sol_drag.fuel_used < sol_nodrag.fuel_used + 5.0,
        "drag should reduce or maintain fuel: {:.2} vs {:.2}",
        sol_drag.fuel_used, sol_nodrag.fuel_used,
    );
}

#[test]
fn test_scvx_virtual_control_vanishes() {
    let sol = solve_scvx(&mars_drag_problem());
    assert!(sol.converged);
    let last = sol.history.last().unwrap();
    assert!(last.vc_norm < 1e-2, "vc_norm={:.2e} did not vanish", last.vc_norm);
}

#[test]
fn test_scvx_defect_small() {
    let sol = solve_scvx(&mars_drag_problem());
    assert!(sol.converged);
    let last = sol.history.last().unwrap();
    assert!(last.defect_pos < 5.0, "pos defect {:.2} too large", last.defect_pos);
    assert!(last.defect_vel < 2.0, "vel defect {:.2} too large", last.defect_vel);
}
