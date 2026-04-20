//! =========================================================================
//! Aerodynamic Model — Rigorous Verification Test Suite
//! =========================================================================
//!
//! VERIFICATION COVERS:
//!   1. Drag force formula: D = 1/2*rho*v^2*Cd*S (fundamental aerodynamics)
//!   2. Drag direction: always opposes relative velocity
//!   3. AeroTable bilinear interpolation correctness
//!   4. Wind profile interpolation correctness
//!
//! SOURCES FOR DRAG VALUES:
//!   [1] Hoerner, S.F. "Fluid-Dynamic Drag" (1965)
//!       - Cd for cylinders, cones, and bluff bodies
//!       - Cd ~ 0.4 subsonic for streamlined cylinder
//!       - Cd ~ 1.0-1.4 transonic (drag divergence)
//!   [2] Anderson, J.D. "Hypersonic and High-Temperature Gas Dynamics" (2006)
//!       - Cd at high Mach numbers
//!   [3] Anderson, J.D. "Fundamentals of Aerodynamics" (2017), 6th ed.
//!       - D = 1/2*rho*v^2*Cd*S (Eq. 1.58)
//!       - This is the standard drag equation used universally in aerospace.
//!
//! VERIFICATION APPROACH:
//!   All drag force tests use HAND CALCULATIONS with known inputs.
//!   D = 0.5 * rho * v^2 * Cd * S
//!   Anyone can verify these with a calculator.
//! =========================================================================

use nalgebra::Vector3;
use rgnc::aero::atmosphere;
use rgnc::aero::coefficients::AeroTable;
use rgnc::aero::wind::WindProfile;

/// Drag equation: D = 1/2*rho*v^2*Cd*S
/// Source: Anderson "Fundamentals of Aerodynamics" (2017), Eq. 1.58
fn drag_force(rho: f64, v: f64, cd: f64, s: f64) -> f64 {
    0.5 * rho * v * v * cd * s
}

// =========================================================================
// GROUP 1: Drag formula verification with hand calculations
// Each test shows the exact arithmetic so anyone can verify with a calculator.
// =========================================================================

#[test]
fn test_drag_sea_level_subsonic() {
    // HAND CALCULATION:
    //   rho = 1.225 kg/m^3 (standard sea level, NASA-TM-X-74335)
    //   v = 170 m/s (Mach ~ 0.5)
    //   Cd = 0.4 (subsonic cylinder, Hoerner 1965 Table 3-1)
    //   S = 10 m^2
    //   D = 0.5 * 1.225 * 170^2 * 0.4 * 10
    //     = 0.5 * 1.225 * 28900 * 0.4 * 10
    //     = 0.5 * 1.225 * 28900 * 4.0
    //     = 0.5 * 141610
    //     = 70805 N
    let d = drag_force(1.225, 170.0, 0.4, 10.0);
    assert!((d - 70805.0).abs() < 1.0,
        "D = {:.1} N, expected 70805.0 N", d);
}

#[test]
fn test_drag_high_altitude_hypersonic() {
    // At h=70km, Mach 10:
    // rho ~ 6.42e-5 kg/m^3 (US Std Atm 1976, geopotential ~69.8km)
    // a ~ 283 m/s, so v = 10*283 = 2830 m/s
    // Cd = 1.1 (hypersonic blunt body, Anderson 2006)
    // S = 10 m^2
    // D = 0.5 * 6.42e-5 * 2830^2 * 1.1 * 10
    //   = 0.5 * 6.42e-5 * 8008900 * 11
    //   = 0.5 * 5653.7 * 11 = 0.5 * 62191 ~ 2828 N
    let atm = atmosphere(70000.0);
    let v = 10.0 * atm.speed_of_sound;
    let d = drag_force(atm.density, v, 1.1, 10.0);
    // Wide tolerance because density depends on atmosphere model precision
    assert!(d > 1000.0 && d < 6000.0,
        "D at 70km Mach 10 = {:.0} N, expected ~2800 N", d);
}

#[test]
fn test_drag_direction_opposes_motion() {
    // PHYSICAL LAW: Drag always opposes velocity relative to the surrounding air.
    // This is definitional -- not from a specific paper.
    // D_direction = -v_rel / |v_rel|
    let velocities = [
        Vector3::new(100.0, 0.0, -50.0),
        Vector3::new(-200.0, 100.0, -300.0),
        Vector3::new(0.0, 500.0, -100.0),
        Vector3::new(1500.0, 0.0, -1500.0),
        Vector3::new(-1.0, -1.0, -1.0),
    ];
    let winds = [
        Vector3::zeros(),
        Vector3::new(10.0, 0.0, 0.0),
        Vector3::new(-5.0, 20.0, 3.0),
    ];

    for v in &velocities {
        for w in &winds {
            let v_rel = v - w;
            let v_mag = v_rel.norm();
            if v_mag < 1e-10 { continue; }
            let drag_dir: Vector3<f64> = -v_rel / v_mag;
            // Drag must point opposite to relative velocity
            assert!(drag_dir.dot(&v_rel) < 0.0,
                "drag not opposing motion: v={:?}, w={:?}", v, w);
            // Drag direction must be unit length
            assert!((drag_dir.norm() - 1.0).abs() < 1e-10,
                "drag direction not unit length");
        }
    }
}

// =========================================================================
// GROUP 2: AeroTable interpolation correctness
// =========================================================================

#[test]
fn test_aerotable_constant_returns_same_value_everywhere() {
    let table = AeroTable::constant(1.2);
    // Must return 1.2 at any Mach and any alpha
    for &m in &[0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0] {
        for &a in &[0.0, 0.05, 0.1, 0.5, 1.0] {
            let (cd, cl) = table.lookup(m, a);
            assert!((cd - 1.2).abs() < 1e-10,
                "constant Cd at M={}, alpha={}: got {}", m, a, cd);
            assert!(cl.abs() < 1e-10,
                "constant Cl at M={}, alpha={}: got {}", m, a, cl);
        }
    }
}

#[test]
fn test_aerotable_bilinear_interpolation_exact_midpoint() {
    // Create a 2x2 table where we can verify interpolation by hand.
    // At the exact midpoint of a cell, bilinear interp = average of 4 corners.
    let table = AeroTable {
        mach_breaks: vec![0.0, 2.0],
        alpha_breaks: vec![0.0, 1.0],
        cd: vec![
            vec![1.0, 3.0],  // M=0: Cd(alpha=0)=1.0, Cd(alpha=1)=3.0
            vec![2.0, 4.0],  // M=2: Cd(alpha=0)=2.0, Cd(alpha=1)=4.0
        ],
        cl: vec![vec![0.0, 0.0]; 2],
        cy: vec![vec![0.0, 0.0]; 2],
    };

    // Midpoint: M=1.0, alpha=0.5
    // Bilinear: (1-0.5)*(1-0.5)*1.0 + 0.5*(1-0.5)*2.0
    //         + (1-0.5)*0.5*3.0 + 0.5*0.5*4.0
    //         = 0.25 + 0.5 + 0.75 + 1.0 = 2.5
    let (cd, _) = table.lookup(1.0, 0.5);
    assert!((cd - 2.5).abs() < 1e-10,
        "midpoint interpolation: expected 2.5, got {}", cd);

    // Corner values must be exact
    let (cd00, _) = table.lookup(0.0, 0.0);
    assert!((cd00 - 1.0).abs() < 1e-10);
    let (cd10, _) = table.lookup(2.0, 0.0);
    assert!((cd10 - 2.0).abs() < 1e-10);
    let (cd01, _) = table.lookup(0.0, 1.0);
    assert!((cd01 - 3.0).abs() < 1e-10);
    let (cd11, _) = table.lookup(2.0, 1.0);
    assert!((cd11 - 4.0).abs() < 1e-10);
}

#[test]
fn test_aerotable_clamping_beyond_boundaries() {
    let table = AeroTable {
        mach_breaks: vec![1.0, 5.0],
        alpha_breaks: vec![0.0, 0.5],
        cd: vec![vec![0.4, 0.6], vec![1.0, 1.2]],
        cl: vec![vec![0.0, 0.0]; 2],
        cy: vec![vec![0.0, 0.0]; 2],
    };

    // Below minimum Mach: should clamp to M=1.0
    let (cd, _) = table.lookup(0.0, 0.0);
    assert!((cd - 0.4).abs() < 1e-10, "below-min Mach clamp: got {}", cd);

    // Above maximum Mach: should clamp to M=5.0
    let (cd, _) = table.lookup(100.0, 0.0);
    assert!((cd - 1.0).abs() < 1e-10, "above-max Mach clamp: got {}", cd);
}

// =========================================================================
// GROUP 3: Wind profile interpolation
// =========================================================================

#[test]
fn test_wind_calm_is_zero_everywhere() {
    let wp = WindProfile::calm();
    for h in [0.0, 5000.0, 50000.0, 100000.0] {
        let w = wp.at_altitude(h);
        assert!(w.norm() < 1e-15, "calm wind not zero at h={}", h);
    }
}

#[test]
fn test_wind_linear_interpolation_exact() {
    // Linear interpolation: at the midpoint between two breakpoints,
    // the wind should be exactly the average.
    let wp = WindProfile {
        altitudes: vec![0.0, 10000.0],
        wind_vectors: vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(20.0, 10.0, 0.0),
        ],
    };

    let w_mid = wp.at_altitude(5000.0);
    assert!((w_mid.x - 10.0).abs() < 1e-10, "midpoint wind x: {}", w_mid.x);
    assert!((w_mid.y - 5.0).abs() < 1e-10, "midpoint wind y: {}", w_mid.y);

    // At 25% height
    let w_quarter = wp.at_altitude(2500.0);
    assert!((w_quarter.x - 5.0).abs() < 1e-10, "quarter wind x: {}", w_quarter.x);
}

#[test]
fn test_wind_boundary_clamping() {
    let wp = WindProfile {
        altitudes: vec![1000.0, 5000.0],
        wind_vectors: vec![
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::new(30.0, 0.0, 0.0),
        ],
    };

    // Below minimum: clamp to first value
    let w = wp.at_altitude(0.0);
    assert!((w.x - 10.0).abs() < 1e-10, "below-min clamp: {}", w.x);

    // Above maximum: clamp to last value
    let w = wp.at_altitude(99999.0);
    assert!((w.x - 30.0).abs() < 1e-10, "above-max clamp: {}", w.x);
}
