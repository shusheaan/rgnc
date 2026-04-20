//! =========================================================================
//! Gurobi MIP Solver — Basic Integration Tests
//! =========================================================================
//!
//! WHAT THIS TESTS:
//!   Verifies that the Gurobi C FFI binding works correctly by solving
//!   small, hand-verifiable MIP problems. These tests confirm that:
//!   1. Environment creation and licensing works
//!   2. Variables (continuous, integer, binary) can be added
//!   3. Linear constraints are correctly communicated to Gurobi
//!   4. The solver finds the correct optimal solution
//!   5. Solution extraction returns correct values
//!
//! WHY THIS MATTERS:
//!   Gurobi is the MIP backend for all robust optimization features:
//!   - Scenario-based chance constraints (binary z_i variables)
//!   - Trajectory library set cover (binary selection variables)
//!   - Safety boundary computation
//!   If ANY of these basic operations fail, all MIP-based features are broken.
//!
//! PREREQUISITE:
//!   These tests require:
//!   1. Gurobi installed (version 13.0.x)
//!   2. GUROBI_HOME environment variable set
//!   3. Valid Gurobi license
//!   If Gurobi is not available, these tests are skipped via #[cfg(feature = "gurobi")].
//!
//! HOW TO VERIFY:
//!   Each test solves a problem small enough to verify by hand or inspection.
//!   Expected solutions are documented inline with the reasoning.
//! =========================================================================

#[cfg(feature = "gurobi")]
mod gurobi_tests {
    use rgnc::gurobi::model::{GrbModel, Status, VarType};

    // =====================================================================
    // TEST 1: Trivial LP — single variable
    // min x,  s.t. x >= 5
    // Optimal: x* = 5, obj* = 5
    // =====================================================================

    #[test]
    fn test_trivial_lp() {
        let mut model = GrbModel::new("trivial_lp")
            .expect("Failed to create Gurobi model — is GUROBI_HOME set and license valid?");

        // One continuous variable x with obj coeff = 1.0 (minimize x)
        // lb = 0, ub = infinity (1e20)
        let x = model.add_var("x", 0.0, 1e20, 1.0, VarType::Continuous).unwrap();

        // Constraint: x >= 5  →  1*x >= 5
        model.add_constr("c1", &[(x, 1.0)], '>', 5.0).unwrap();

        model.set_minimize(true).unwrap();
        let status = model.optimize().unwrap();

        assert_eq!(status, Status::Optimal,
            "trivial LP should be optimal");

        let obj = model.obj_val().unwrap();
        assert!((obj - 5.0).abs() < 1e-6,
            "optimal objective = {}, expected 5.0", obj);

        let sol = model.get_solution().unwrap();
        assert!((sol[x] - 5.0).abs() < 1e-6,
            "x* = {}, expected 5.0", sol[x]);
    }

    // =====================================================================
    // TEST 2: Small LP — 2 variables, 2 constraints
    // max 3x + 2y
    // s.t. x + y <= 10
    //      x - y <= 4
    //      x, y >= 0
    //
    // HAND SOLUTION:
    //   Binding constraints: x + y = 10, x - y = 4
    //   → x = 7, y = 3
    //   → obj = 3*7 + 2*3 = 27
    // =====================================================================

    #[test]
    fn test_small_lp_two_variables() {
        let mut model = GrbModel::new("small_lp").unwrap();

        let x = model.add_var("x", 0.0, 1e20, 3.0, VarType::Continuous).unwrap();
        let y = model.add_var("y", 0.0, 1e20, 2.0, VarType::Continuous).unwrap();

        // x + y <= 10
        model.add_constr("c1", &[(x, 1.0), (y, 1.0)], '<', 10.0).unwrap();
        // x - y <= 4
        model.add_constr("c2", &[(x, 1.0), (y, -1.0)], '<', 4.0).unwrap();

        model.set_minimize(false).unwrap(); // maximize
        let status = model.optimize().unwrap();

        assert_eq!(status, Status::Optimal);

        let obj = model.obj_val().unwrap();
        assert!((obj - 27.0).abs() < 1e-6,
            "max obj = {}, expected 27.0", obj);

        let sol = model.get_solution().unwrap();
        assert!((sol[x] - 7.0).abs() < 1e-6, "x* = {}, expected 7.0", sol[x]);
        assert!((sol[y] - 3.0).abs() < 1e-6, "y* = {}, expected 3.0", sol[y]);
    }

    // =====================================================================
    // TEST 3: Integer Program — binary knapsack
    // max 5x1 + 4x2 + 3x3
    // s.t. 2x1 + 3x2 + 2x3 <= 5
    //      x1, x2, x3 ∈ {0, 1}
    //
    // HAND SOLUTION:
    //   Enumerate all 8 combinations:
    //   (1,0,1): 2+2=4 ≤ 5, obj = 5+3 = 8 ← optimal
    //   (1,1,0): 2+3=5 ≤ 5, obj = 5+4 = 9 ← OPTIMAL
    //   (0,1,1): 3+2=5 ≤ 5, obj = 4+3 = 7
    //   (1,1,1): 2+3+2=7 > 5, infeasible
    //   Best: x1=1, x2=1, x3=0, obj=9
    // =====================================================================

    #[test]
    fn test_binary_knapsack() {
        let mut model = GrbModel::new("knapsack").unwrap();

        let x1 = model.add_var("x1", 0.0, 1.0, 5.0, VarType::Binary).unwrap();
        let x2 = model.add_var("x2", 0.0, 1.0, 4.0, VarType::Binary).unwrap();
        let x3 = model.add_var("x3", 0.0, 1.0, 3.0, VarType::Binary).unwrap();

        // 2x1 + 3x2 + 2x3 <= 5
        model.add_constr("weight", &[(x1, 2.0), (x2, 3.0), (x3, 2.0)], '<', 5.0).unwrap();

        model.set_minimize(false).unwrap(); // maximize
        let status = model.optimize().unwrap();

        assert_eq!(status, Status::Optimal);

        let obj = model.obj_val().unwrap();
        assert!((obj - 9.0).abs() < 1e-6,
            "knapsack obj = {}, expected 9.0", obj);

        let sol = model.get_solution().unwrap();
        assert!((sol[x1] - 1.0).abs() < 1e-6, "x1* = {}, expected 1", sol[x1]);
        assert!((sol[x2] - 1.0).abs() < 1e-6, "x2* = {}, expected 1", sol[x2]);
        assert!(sol[x3].abs() < 1e-6, "x3* = {}, expected 0", sol[x3]);
    }

    // =====================================================================
    // TEST 4: Mixed-Integer Program — continuous + integer variables
    // This is the type of problem used in chance-constrained optimization:
    // binary z_i selects which scenarios to discard.
    //
    // min x + 10*z
    // s.t. x + 3 >= 5 * (1 - z)   →   x + 5z >= 2
    //      x >= 0, z ∈ {0, 1}
    //
    // If z=0: x >= 2, obj = 2 + 0 = 2
    // If z=1: x >= 0 (relaxed via big-M equivalent), obj = 0 + 10 = 10
    // Optimal: z=0, x=2, obj=2
    // =====================================================================

    #[test]
    fn test_mixed_integer_program() {
        let mut model = GrbModel::new("mip").unwrap();

        let x = model.add_var("x", 0.0, 1e20, 1.0, VarType::Continuous).unwrap();
        let z = model.add_var("z", 0.0, 1.0, 10.0, VarType::Binary).unwrap();

        // x + 5z >= 2
        model.add_constr("scenario", &[(x, 1.0), (z, 5.0)], '>', 2.0).unwrap();

        model.set_minimize(true).unwrap();
        let status = model.optimize().unwrap();

        assert_eq!(status, Status::Optimal);

        let obj = model.obj_val().unwrap();
        assert!((obj - 2.0).abs() < 1e-6,
            "MIP obj = {}, expected 2.0", obj);

        let sol = model.get_solution().unwrap();
        assert!((sol[x] - 2.0).abs() < 1e-6, "x* = {}, expected 2.0", sol[x]);
        assert!(sol[z].abs() < 1e-6, "z* = {}, expected 0", sol[z]);
    }

    // =====================================================================
    // TEST 5: Infeasible model detection
    // min x,  s.t. x >= 10, x <= 5  → infeasible
    // =====================================================================

    #[test]
    fn test_infeasible_detection() {
        let mut model = GrbModel::new("infeasible").unwrap();

        let x = model.add_var("x", 0.0, 1e20, 1.0, VarType::Continuous).unwrap();

        model.add_constr("lb", &[(x, 1.0)], '>', 10.0).unwrap();
        model.add_constr("ub", &[(x, 1.0)], '<', 5.0).unwrap();

        model.set_minimize(true).unwrap();
        let status = model.optimize().unwrap();

        assert_eq!(status, Status::Infeasible,
            "contradictory constraints should be detected as infeasible");
    }
}
