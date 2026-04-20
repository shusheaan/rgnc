//! Clarabel SOCP solver interface.
//!
//! Wraps Clarabel's API for solving second-order cone programs
//! in the standard form:
//!   min  (1/2)x'Px + q'x
//!   s.t. Ax + s = b,  s ∈ K
//!
//! where K is a product of ZeroCone, NonnegativeCone, and SecondOrderCone.

use crate::solver::result::{SolveResult, SolveStatusGeneric};
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, SolverStatus,
    SupportedConeT,
};
use std::time::Instant;

/// SOCP problem data in Clarabel's standard conic form.
pub struct SocpData {
    pub p: CscMatrix<f64>,
    pub q: Vec<f64>,
    pub a: CscMatrix<f64>,
    pub b: Vec<f64>,
    pub cones: Vec<SupportedConeT<f64>>,
}

/// Solve an SOCP problem using Clarabel.
pub fn solve_socp(data: &SocpData, verbose: bool) -> SolveResult {
    let mut settings = DefaultSettings::default();
    settings.verbose = verbose;

    let start = Instant::now();
    let mut solver =
        DefaultSolver::new(&data.p, &data.q, &data.a, &data.b, &data.cones, settings);
    solver.solve();
    let elapsed = start.elapsed();

    let status = match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => SolveStatusGeneric::Solved,
        SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => {
            SolveStatusGeneric::Infeasible
        }
        SolverStatus::DualInfeasible | SolverStatus::AlmostDualInfeasible => {
            SolveStatusGeneric::DualInfeasible
        }
        SolverStatus::MaxIterations | SolverStatus::MaxTime => SolveStatusGeneric::MaxIterations,
        SolverStatus::NumericalError | SolverStatus::InsufficientProgress => {
            SolveStatusGeneric::NumericalError
        }
        _ => SolveStatusGeneric::Unknown,
    };

    SolveResult {
        objective: solver.solution.obj_val,
        primal: solver.solution.x.clone(),
        status,
        solve_time: elapsed,
        iterations: solver.solution.iterations as usize,
    }
}
