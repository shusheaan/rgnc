//! =========================================================================
//! Scenario Struct — Verification Tests
//! =========================================================================
//!
//! WHAT THIS TESTS:
//!   The `Scenario` struct in `src/robust/scenario.rs` represents a single
//!   uncertainty realization for robust optimization. Each scenario contains:
//!   - wind_profile: altitude-dependent wind vectors
//!   - density_factor: multiplier on standard atmosphere density
//!   - cd_factor: multiplier on nominal drag coefficient
//!   - thrust_bias: additive error on engine thrust (N)
//!
//! WHY THIS MATTERS:
//!   The nominal scenario (all factors = 1.0, no wind, no thrust bias) is
//!   the baseline against which all robust methods are compared. If the
//!   nominal scenario has any perturbation, the entire comparison framework
//!   is invalid — robust method would appear to give zero improvement.
//!
//! HOW TO VERIFY:
//!   Nominal scenario must have:
//!   - density_factor = 1.0 exactly (no atmosphere perturbation)
//!   - cd_factor = 1.0 exactly (no aero coefficient perturbation)
//!   - thrust_bias = 0.0 exactly (no engine error)
//!   - wind = zero at all altitudes
//! =========================================================================

use rgnc::robust::Scenario;

#[test]
fn test_nominal_scenario() {
    // WHAT: Verify Scenario::nominal() produces a zero-perturbation baseline.
    // WHY: If ANY factor deviates from 1.0 (or 0.0 for bias), the "nominal"
    //       trajectory will be perturbed, making all robust comparisons invalid.
    let s = Scenario::nominal(0);

    // ID must match what was passed in
    assert_eq!(s.id, 0);

    // All perturbation factors must be exactly 1.0 (identity multiplier)
    assert!((s.density_factor - 1.0).abs() < 1e-15,
        "nominal density_factor must be exactly 1.0, got {}", s.density_factor);
    assert!((s.cd_factor - 1.0).abs() < 1e-15,
        "nominal cd_factor must be exactly 1.0, got {}", s.cd_factor);

    // Thrust bias must be exactly 0.0 (no engine error)
    assert!((s.thrust_bias).abs() < 1e-15,
        "nominal thrust_bias must be exactly 0.0, got {}", s.thrust_bias);

    // Wind profile must have at least 2 points (for interpolation)
    assert_eq!(s.wind_profile.len(), 2,
        "wind profile should have 2 altitude points for interpolation");
}

#[test]
fn test_nominal_scenario_wind_is_zero() {
    // WHAT: Verify that the nominal wind profile is zero everywhere.
    // WHY: Any non-zero wind in the nominal scenario would make the
    //       "nominal trajectory" already account for wind, defeating the
    //       purpose of the robust optimization comparison.
    let s = Scenario::nominal(42);
    for &(alt, ref w) in &s.wind_profile {
        assert!(w.norm() < 1e-15,
            "nominal wind must be zero at alt={:.0} m, got norm={:.2e}",
            alt, w.norm());
    }
}
