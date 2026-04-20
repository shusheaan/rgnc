//! =========================================================================
//! Gravity Model Verification Tests
//! =========================================================================
//!
//! SOURCES:
//!   [1] WGS-84 Earth Gravitational Model
//!       g0 = 9.80665 m/s² (standard gravitational acceleration)
//!       This is a DEFINED constant, not measured.
//!       Source: 3rd CGPM (1901), Resolution 2
//!
//!   [2] Inverse-square law:
//!       g(h) = g0 * (Re / (Re + h))²
//!       where Re = 6,371,000 m (mean Earth radius)
//!       This is Newtonian gravity for a spherical Earth.
//!       Source: Any introductory physics textbook, e.g.,
//!       Halliday, Resnick & Walker, "Fundamentals of Physics"
//!
//! VERIFICATION:
//!   The constant gravity model returns exactly g0.
//!   The altitude-dependent model follows the inverse-square law exactly.
//!   Both can be verified by hand calculation.
//! =========================================================================

use rgnc::dynamics::gravity::{gravity_constant, gravity_altitude};

const G0: f64 = 9.80665;
const RE: f64 = 6_371_000.0;

#[test]
fn test_gravity_constant_value() {
    // g0 = 9.80665 m/s² is exact by definition (3rd CGPM 1901)
    let g = gravity_constant();
    assert!((g.x).abs() < 1e-15, "gx should be zero");
    assert!((g.y).abs() < 1e-15, "gy should be zero");
    assert!((g.z - (-G0)).abs() < 1e-10,
        "gz should be -{}: got {}", G0, g.z);
}

#[test]
fn test_gravity_constant_direction() {
    // Gravity must point in -z direction (toward Earth center)
    let g = gravity_constant();
    assert!(g.z < 0.0, "gravity must point downward (negative z)");
    assert!(g.x == 0.0 && g.y == 0.0, "gravity has no horizontal component");
}

#[test]
fn test_gravity_altitude_sea_level() {
    // At h=0, altitude-dependent gravity should equal g0
    let g = gravity_altitude(0.0);
    assert!((g.z - (-G0)).abs() < 1e-10,
        "gravity at sea level: expected {}, got {}", -G0, g.z);
}

#[test]
fn test_gravity_altitude_iss() {
    // ISS altitude ~400 km
    // g(400km) = 9.80665 * (6371000/6771000)^2 = 8.693 m/s²
    // Hand calculation: (6371/6771)^2 = 0.88435, g = 8.671 m/s²
    let h = 400_000.0;
    let expected = G0 * (RE / (RE + h)).powi(2);
    let g = gravity_altitude(h);
    assert!((g.z - (-expected)).abs() < 1e-6,
        "gravity at ISS altitude: expected {:.4}, got {:.4}", -expected, g.z);
    // Gravity at ISS should be ~88% of sea level
    assert!(g.z.abs() > 8.0 && g.z.abs() < 9.0,
        "g at 400km should be ~8.7 m/s²: got {:.3}", g.z.abs());
}

#[test]
fn test_gravity_altitude_high() {
    // At 1 Earth radius above surface (h = Re = 6371 km):
    // g = g0 * (1/2)^2 = g0/4 = 2.4517 m/s²
    let h = RE;
    let expected = G0 / 4.0;
    let g = gravity_altitude(h);
    assert!((g.z - (-expected)).abs() < 1e-6,
        "gravity at h=Re: expected {:.4}, got {:.4}", -expected, g.z);
}

#[test]
fn test_gravity_decreases_with_altitude() {
    // g must decrease monotonically with altitude
    let mut prev_g = f64::MAX;
    for h_km in 0..200 {
        let h = h_km as f64 * 1000.0;
        let g = gravity_altitude(h);
        assert!(g.z.abs() < prev_g,
            "gravity not decreasing at h={:.0} km", h_km);
        prev_g = g.z.abs();
    }
}

#[test]
fn test_gravity_inverse_square_formula() {
    // Verify g(h) = g0 * (Re/(Re+h))^2 for multiple altitudes
    for &h in &[0.0, 1000.0, 10000.0, 100000.0, 1000000.0] {
        let g = gravity_altitude(h);
        let expected = -G0 * (RE / (RE + h)).powi(2);
        assert!((g.z - expected).abs() < 1e-10,
            "inverse-square law at h={:.0}: expected {:.6}, got {:.6}",
            h, expected, g.z);
    }
}
