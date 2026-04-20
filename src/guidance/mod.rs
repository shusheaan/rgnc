pub mod formulation;
pub mod linearize;
pub mod pdg;
pub mod scvx;
pub mod scvx_formulation;
pub mod trust_region;

pub use pdg::{PdgProblem, PdgSolution, SolveStatus, solve_pdg_2d, solve_pdg_2d_free_tf, solve_pdg_3d, solve_pdg_3d_free_tf};
pub use scvx::{ScvxConfig, ScvxProblem, ScvxSolution, ScvxIteration, solve_scvx};
pub use trust_region::TrustRegion;
