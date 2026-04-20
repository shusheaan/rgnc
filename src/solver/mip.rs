/// MIP solver trait and implementations.
///
/// Provides a trait-based abstraction over MIP solvers
/// to support both HiGHS (open-source) and Gurobi (commercial).

/// Result from a MIP solve.
#[derive(Debug, Clone)]
pub struct MipResult {
    pub objective: f64,
    pub solution: Vec<f64>,
    pub status: MipStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MipStatus {
    Optimal,
    Infeasible,
    Unbounded,
    TimeLimit,
    Error,
}

/// Trait for MIP solver backends.
pub trait MipSolver {
    fn add_variable(&mut self, lb: f64, ub: f64, obj: f64, is_integer: bool) -> usize;
    fn add_constraint(&mut self, coeffs: &[(usize, f64)], lb: f64, ub: f64);
    fn solve(&mut self) -> MipResult;
}
