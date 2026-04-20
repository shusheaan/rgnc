//! Safe Rust wrapper around the Gurobi C API for MIP solving.
//!
//! Provides a minimal `GrbModel` that can:
//! 1. Create an environment and model
//! 2. Add continuous, integer, and binary variables
//! 3. Add linear constraints
//! 4. Set a linear objective
//! 5. Solve and extract the solution
//!
//! This is intentionally minimal — only what's needed for trajectory
//! optimization MIP formulations (set cover, chance constraints).
//!
//! Adapted from `star` project's `src/core/grb.rs`, stripped of
//! portfolio-specific features (multi-objective, quadratic constraints, etc.)
//!
//! # Example
//!
//! ```ignore
//! let mut model = GrbModel::new("my_mip")?;
//! let x = model.add_var("x", 0.0, 10.0, 1.0, VarType::Continuous)?;
//! let y = model.add_var("y", 0.0, 10.0, 2.0, VarType::Integer)?;
//! model.add_constr("c1", &[(x, 1.0), (y, 1.0)], '<', 8.0)?;
//! let status = model.optimize()?;
//! let solution = model.get_solution()?;
//! ```

use super::ffi::*;
use anyhow::{bail, Result};
use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

/// Variable type for Gurobi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarType {
    Continuous, // 'C' = 67
    Binary,     // 'B' = 66
    Integer,    // 'I' = 73
}

impl VarType {
    fn to_char(self) -> i8 {
        match self {
            VarType::Continuous => b'C' as i8,
            VarType::Binary => b'B' as i8,
            VarType::Integer => b'I' as i8,
        }
    }
}

/// Gurobi optimization status codes (subset).
/// Full list: https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Status {
    Optimal,      // 2
    Infeasible,   // 3
    Unbounded,    // 5
    TimeLimit,    // 9
    Other(i32),   // anything else
}

impl From<i32> for Status {
    fn from(code: i32) -> Self {
        match code {
            2 => Status::Optimal,
            3 => Status::Infeasible,
            5 => Status::Unbounded,
            9 => Status::TimeLimit,
            x => Status::Other(x),
        }
    }
}

/// Minimal Gurobi model wrapper. Owns the environment and model pointers.
///
/// On drop, frees the model and environment in the correct order.
pub struct GrbModel {
    model: *mut GRBmodel,
    env: *mut GRBenv,
    n_vars: usize,
}

impl Drop for GrbModel {
    fn drop(&mut self) {
        unsafe {
            if !self.model.is_null() {
                GRBfreemodel(self.model);
            }
            if !self.env.is_null() {
                GRBfreeenv(self.env);
            }
        }
    }
}

impl GrbModel {
    /// Create a new empty Gurobi model.
    ///
    /// This initializes the Gurobi environment, suppresses console output,
    /// and creates an empty model ready for variables and constraints.
    pub fn new(name: &str) -> Result<Self> {
        let mut env: *mut GRBenv = ptr::null_mut();
        let mut model: *mut GRBmodel = ptr::null_mut();

        unsafe {
            // Create environment with version check
            check(
                GRBemptyenvinternal(
                    &mut env,
                    GRB_VERSION_MAJOR as i32,
                    GRB_VERSION_MINOR as i32,
                    GRB_VERSION_TECHNICAL as i32,
                ),
                env,
            )?;

            // Suppress console output by default
            let param = CString::new("OutputFlag")?;
            check(GRBsetintparam(env, param.as_ptr(), 0), env)?;

            // Start the environment (acquires license)
            check(GRBstartenv(env), env)?;

            // Create empty model
            let name_c = CString::new(name)?;
            check(
                GRBnewmodel(
                    env,
                    &mut model,
                    name_c.as_ptr(),
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                env,
            )?;
        }

        Ok(GrbModel {
            model,
            env,
            n_vars: 0,
        })
    }

    /// Add a variable to the model.
    ///
    /// Returns the variable index (0-based), used to reference this variable
    /// in constraints and objective.
    ///
    /// # Arguments
    /// - `name`: variable name (for debugging/LP output)
    /// - `lb`: lower bound
    /// - `ub`: upper bound
    /// - `obj`: objective coefficient
    /// - `vtype`: Continuous, Integer, or Binary
    pub fn add_var(
        &mut self,
        name: &str,
        lb: f64,
        ub: f64,
        obj: f64,
        vtype: VarType,
    ) -> Result<usize> {
        let name_c = CString::new(name)?;
        let idx = self.n_vars;

        unsafe {
            check(
                GRBaddvar(
                    self.model,
                    0,                    // no column entries
                    ptr::null_mut(),      // column indices
                    ptr::null_mut(),      // column values
                    obj,                  // objective coefficient
                    lb,
                    ub,
                    vtype.to_char(),
                    name_c.as_ptr(),
                ),
                self.env,
            )?;
        }

        self.n_vars += 1;
        Ok(idx)
    }

    /// Add a linear constraint: sum(coeff[i] * var[i]) sense rhs
    ///
    /// # Arguments
    /// - `name`: constraint name
    /// - `terms`: slice of (variable_index, coefficient) pairs
    /// - `sense`: '<' (<=), '>' (>=), or '=' (==)
    /// - `rhs`: right-hand side value
    pub fn add_constr(
        &mut self,
        name: &str,
        terms: &[(usize, f64)],
        sense: char,
        rhs: f64,
    ) -> Result<()> {
        let name_c = CString::new(name)?;
        let mut ind: Vec<c_int> = terms.iter().map(|&(i, _)| i as c_int).collect();
        let mut val: Vec<f64> = terms.iter().map(|&(_, v)| v).collect();
        let sense_byte = sense as i8;

        unsafe {
            check(
                GRBaddconstr(
                    self.model,
                    ind.len() as c_int,
                    ind.as_mut_ptr(),
                    val.as_mut_ptr(),
                    sense_byte,
                    rhs,
                    name_c.as_ptr(),
                ),
                self.env,
            )?;
        }

        Ok(())
    }

    /// Set the optimization direction.
    /// `minimize = true` → minimize objective, `false` → maximize.
    pub fn set_minimize(&mut self, minimize: bool) -> Result<()> {
        let attr = CString::new("ModelSense")?;
        let sense = if minimize { 1 } else { -1 };
        unsafe {
            check(
                GRBsetintattr(self.model, attr.as_ptr(), sense),
                self.env,
            )?;
        }
        Ok(())
    }

    /// Solve the model. Returns the optimization status.
    pub fn optimize(&mut self) -> Result<Status> {
        unsafe {
            check(GRBupdatemodel(self.model), self.env)?;
            check(GRBoptimize(self.model), self.env)?;
        }

        let mut status: i32 = 0;
        let attr = CString::new("Status")?;
        unsafe {
            check(
                GRBgetintattr(self.model, attr.as_ptr(), &mut status),
                self.env,
            )?;
        }

        Ok(Status::from(status))
    }

    /// Get the optimal objective value (only valid after Status::Optimal).
    pub fn obj_val(&self) -> Result<f64> {
        let mut val: f64 = 0.0;
        let attr = CString::new("ObjVal")?;
        unsafe {
            check(
                GRBgetdblattr(self.model, attr.as_ptr(), &mut val),
                self.env,
            )?;
        }
        Ok(val)
    }

    /// Get all variable solution values (only valid after Status::Optimal).
    pub fn get_solution(&self) -> Result<Vec<f64>> {
        let n = self.n_vars as c_int;
        let mut x = vec![0.0_f64; self.n_vars];
        let attr = CString::new("X")?;
        unsafe {
            check(
                GRBgetdblattrarray(self.model, attr.as_ptr(), 0, n, x.as_mut_ptr()),
                self.env,
            )?;
        }
        Ok(x)
    }

    /// Number of variables in the model.
    pub fn num_vars(&self) -> usize {
        self.n_vars
    }

    /// Enable or disable console output.
    pub fn set_output(&mut self, enable: bool) -> Result<()> {
        let param = CString::new("OutputFlag")?;
        let model_env = unsafe { GRBgetenv(self.model) };
        unsafe {
            check(
                GRBsetintparam(model_env, param.as_ptr(), if enable { 1 } else { 0 }),
                model_env,
            )?;
        }
        Ok(())
    }

    /// Set the time limit for optimization (seconds).
    pub fn set_time_limit(&mut self, seconds: f64) -> Result<()> {
        let param = CString::new("TimeLimit")?;
        let model_env = unsafe { GRBgetenv(self.model) };
        unsafe {
            check(
                GRBsetdblparam(model_env, param.as_ptr(), seconds),
                model_env,
            )?;
        }
        Ok(())
    }
}

/// Convert a non-zero Gurobi error code into an anyhow error.
/// Reads the human-readable message from GRBgeterrormsg.
fn check(error: c_int, env: *mut GRBenv) -> Result<()> {
    if error == 0 {
        return Ok(());
    }
    let msg = if !env.is_null() {
        let ptr = unsafe { GRBgeterrormsg(env) };
        if !ptr.is_null() {
            unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        } else {
            "unknown error".into()
        }
    } else {
        format!("Gurobi error code {}", error)
    };
    bail!("Gurobi error {}: {}", error, msg)
}
