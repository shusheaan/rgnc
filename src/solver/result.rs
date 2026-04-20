/// Unified solve result types shared across solver backends.

use std::time::Duration;

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub objective: f64,
    pub primal: Vec<f64>,
    pub status: SolveStatusGeneric,
    pub solve_time: Duration,
    pub iterations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolveStatusGeneric {
    Solved,
    Infeasible,
    DualInfeasible,
    MaxIterations,
    NumericalError,
    Unknown,
}
