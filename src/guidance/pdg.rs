use crate::dynamics::{Control, State};
use crate::guidance::formulation::{build_pdg_2d, build_pdg_3d};
use crate::solver::socp::solve_socp;
use crate::vehicle::VehicleParams;
use nalgebra::{UnitQuaternion, Vector3};
use std::time::Duration;

/// Powered Descent Guidance problem definition.
#[derive(Debug, Clone)]
pub struct PdgProblem {
    pub vehicle: VehicleParams,
    pub initial_state: State,
    pub target_state: State,
    pub n_timesteps: usize,
    pub dt: f64,
    /// Gravitational acceleration magnitude (m/s², positive value).
    /// Mars: 3.7114, Earth: 9.80665
    pub gravity: f64,
}

/// Status of an optimization solve.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveStatus {
    Optimal,
    Infeasible,
    MaxIterations,
    NumericalError,
}

/// Solution from the PDG solver.
#[derive(Debug, Clone)]
pub struct PdgSolution {
    pub trajectory: Vec<State>,
    pub controls: Vec<Control>,
    pub fuel_used: f64,
    pub solve_time: Duration,
    pub status: SolveStatus,
}

/// Solve the 2D PDG problem using SOCP (Açıkmeşe 2007 formulation).
///
/// Returns fuel-optimal trajectory from initial_state to target_state
/// under constant gravity, no atmosphere.
pub fn solve_pdg_2d(problem: &PdgProblem, verbose: bool) -> PdgSolution {
    let data = build_pdg_2d(problem);
    let result = solve_socp(&data, verbose);

    let status = match result.status {
        crate::solver::result::SolveStatusGeneric::Solved => SolveStatus::Optimal,
        crate::solver::result::SolveStatusGeneric::Infeasible => SolveStatus::Infeasible,
        crate::solver::result::SolveStatusGeneric::MaxIterations => SolveStatus::MaxIterations,
        _ => SolveStatus::NumericalError,
    };

    if status != SolveStatus::Optimal {
        return PdgSolution {
            trajectory: vec![],
            controls: vec![],
            fuel_used: 0.0,
            solve_time: result.solve_time,
            status,
        };
    }

    // Extract solution using same index layout as formulation
    let n = problem.n_timesteps;
    let ctrl_off = 4 * (n + 1);
    let mass_off = ctrl_off + 3 * n;
    let x = &result.primal;

    let m_wet = problem.vehicle.dry_mass + problem.vehicle.fuel_mass;
    let identity = UnitQuaternion::identity();

    let mut trajectory = Vec::with_capacity(n + 1);
    for k in 0..=n {
        let z_k = x[mass_off + k];
        trajectory.push(State::new(
            Vector3::new(x[4 * k], 0.0, x[4 * k + 1]),     // pos: (r1, 0, r2)
            Vector3::new(x[4 * k + 2], 0.0, x[4 * k + 3]), // vel: (v1, 0, v2)
            identity,
            Vector3::zeros(),
            z_k.exp(), // mass = e^z
            k as f64 * problem.dt,
        ));
    }

    // Lossless convexification post-processing:
    // Interior-point solvers leave tiny nonzero σ where the engine should be off.
    // Snap σ below threshold to exactly 0. Threshold = T_min/m_wet * 0.01
    let sigma_threshold = problem.vehicle.thrust_min / m_wet * 0.01;

    let mut controls = Vec::with_capacity(n);
    for k in 0..n {
        let u1 = x[ctrl_off + 3 * k];
        let _u2 = x[ctrl_off + 3 * k + 1];
        let raw_sigma = x[ctrl_off + 3 * k + 2];
        let sigma = if raw_sigma < sigma_threshold { 0.0 } else { raw_sigma };
        let mass_k = trajectory[k].mass;

        let thrust = sigma * mass_k;
        let throttle = if sigma < sigma_threshold {
            0.0
        } else {
            ((thrust - problem.vehicle.thrust_min)
                / (problem.vehicle.thrust_max - problem.vehicle.thrust_min))
                .clamp(0.0, 1.0)
        };

        let gimbal_y = if sigma > sigma_threshold {
            -(u1 / sigma).asin().clamp(-problem.vehicle.gimbal_max, problem.vehicle.gimbal_max)
        } else {
            0.0
        };

        controls.push(Control::new(throttle, gimbal_y, 0.0));
    }

    let fuel_used = m_wet - trajectory.last().unwrap().mass;

    PdgSolution {
        trajectory,
        controls,
        fuel_used,
        solve_time: result.solve_time,
        status,
    }
}

/// Solve the 2D PDG problem with free final time.
///
/// Uses golden-section search over dt (with fixed N) to find
/// the fuel-optimal flight time tf = N * dt.
pub fn solve_pdg_2d_free_tf(
    problem: &PdgProblem,
    dt_range: (f64, f64),
    verbose: bool,
) -> PdgSolution {
    let (mut a, mut b) = dt_range;
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate
    let tol = 1e-3; // dt tolerance

    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);

    let eval = |dt: f64| -> f64 {
        let mut p = problem.clone();
        p.dt = dt;
        let sol = solve_pdg_2d(&p, false);
        if sol.status == SolveStatus::Optimal {
            sol.fuel_used
        } else {
            f64::INFINITY
        }
    };

    let mut fc = eval(c);
    let mut fd = eval(d);

    while (b - a) > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = eval(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = eval(d);
        }
    }

    let best_dt = (a + b) / 2.0;
    let mut best_problem = problem.clone();
    best_problem.dt = best_dt;
    let mut sol = solve_pdg_2d(&best_problem, verbose);

    // Update times in trajectory to reflect optimized dt
    for (k, state) in sol.trajectory.iter_mut().enumerate() {
        state.time = k as f64 * best_dt;
    }
    sol
}

/// Solve the 3D PDG problem using SOCP (Açıkmeşe 2007 formulation, extended to 3D).
pub fn solve_pdg_3d(problem: &PdgProblem, verbose: bool) -> PdgSolution {
    let data = build_pdg_3d(problem);
    let result = solve_socp(&data, verbose);

    let status = match result.status {
        crate::solver::result::SolveStatusGeneric::Solved => SolveStatus::Optimal,
        crate::solver::result::SolveStatusGeneric::Infeasible => SolveStatus::Infeasible,
        crate::solver::result::SolveStatusGeneric::MaxIterations => SolveStatus::MaxIterations,
        _ => SolveStatus::NumericalError,
    };

    if status != SolveStatus::Optimal {
        return PdgSolution {
            trajectory: vec![],
            controls: vec![],
            fuel_used: 0.0,
            solve_time: result.solve_time,
            status,
        };
    }

    let n = problem.n_timesteps;
    let ctrl_off = 6 * (n + 1);
    let mass_off = ctrl_off + 4 * n;
    let x = &result.primal;

    let m_wet = problem.vehicle.dry_mass + problem.vehicle.fuel_mass;
    let identity = UnitQuaternion::identity();

    let mut trajectory = Vec::with_capacity(n + 1);
    for k in 0..=n {
        let z_k = x[mass_off + k];
        trajectory.push(State::new(
            Vector3::new(x[6 * k], x[6 * k + 1], x[6 * k + 2]),
            Vector3::new(x[6 * k + 3], x[6 * k + 4], x[6 * k + 5]),
            identity,
            Vector3::zeros(),
            z_k.exp(),
            k as f64 * problem.dt,
        ));
    }

    // Lossless convexification post-processing
    let sigma_threshold = problem.vehicle.thrust_min / m_wet * 0.01;

    let mut controls = Vec::with_capacity(n);
    for k in 0..n {
        let u1 = x[ctrl_off + 4 * k];
        let u2 = x[ctrl_off + 4 * k + 1];
        let _u3 = x[ctrl_off + 4 * k + 2];
        let raw_sigma = x[ctrl_off + 4 * k + 3];
        let sigma = if raw_sigma < sigma_threshold { 0.0 } else { raw_sigma };
        let mass_k = trajectory[k].mass;

        let thrust = sigma * mass_k;
        let throttle = if sigma < sigma_threshold {
            0.0
        } else {
            ((thrust - problem.vehicle.thrust_min)
                / (problem.vehicle.thrust_max - problem.vehicle.thrust_min))
                .clamp(0.0, 1.0)
        };

        // Gimbal from u1, u2 components (3D: two gimbal axes)
        let gimbal_y = if sigma > sigma_threshold {
            -(u1 / sigma).asin().clamp(-problem.vehicle.gimbal_max, problem.vehicle.gimbal_max)
        } else {
            0.0
        };
        let gimbal_z = if sigma > sigma_threshold {
            -(u2 / sigma).asin().clamp(-problem.vehicle.gimbal_max, problem.vehicle.gimbal_max)
        } else {
            0.0
        };

        controls.push(Control::new(throttle, gimbal_y, gimbal_z));
    }

    let fuel_used = m_wet - trajectory.last().unwrap().mass;

    PdgSolution {
        trajectory,
        controls,
        fuel_used,
        solve_time: result.solve_time,
        status,
    }
}

/// Solve the 3D PDG problem with free final time.
pub fn solve_pdg_3d_free_tf(
    problem: &PdgProblem,
    dt_range: (f64, f64),
    verbose: bool,
) -> PdgSolution {
    let (mut a, mut b) = dt_range;
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
    let tol = 1e-3;

    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);

    let eval = |dt: f64| -> f64 {
        let mut p = problem.clone();
        p.dt = dt;
        let sol = solve_pdg_3d(&p, false);
        if sol.status == SolveStatus::Optimal {
            sol.fuel_used
        } else {
            f64::INFINITY
        }
    };

    let mut fc = eval(c);
    let mut fd = eval(d);

    while (b - a) > tol {
        if fc < fd {
            b = d; d = c; fd = fc;
            c = b - gr * (b - a);
            fc = eval(c);
        } else {
            a = c; c = d; fc = fd;
            d = a + gr * (b - a);
            fd = eval(d);
        }
    }

    let best_dt = (a + b) / 2.0;
    let mut best_problem = problem.clone();
    best_problem.dt = best_dt;
    let mut sol = solve_pdg_3d(&best_problem, verbose);

    for (k, state) in sol.trajectory.iter_mut().enumerate() {
        state.time = k as f64 * best_dt;
    }
    sol
}
